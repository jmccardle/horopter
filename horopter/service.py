# SPDX-License-Identifier: MIT
"""The request handler behind every route, usable in-process: Service(student).decide(request) -> response dict.

horopter.server wraps it in HTTP; horopter.local() returns one for a model loaded in this process, so code written
against horopter.connect(url) (the HTTP client) runs unchanged on a local model. Needs the torch backend.
"""
import hashlib
import json
import threading
import time

from .api import DecideRequest

CONTRACT = "3.1.0+contract2"
CLAIRIC_NAMES = ("clairic", "@cf/cloudflare/clairic")
NO_IMAGES = ("image input is not supported by this model yet (Jev `assets`, Clairic `images`, box questions); "
             "send text only")
_FIELD = "␟field:"   # hidden question ids for fields (a separator character no client id uses)


def render_state(state):
    """One canonical renderer for training and serving: strings as is, JSON sorted and compact."""
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True, separators=(",", ":"))


def check_question(qid, q):
    t = q.get("type")
    if t == "box":
        return f"question {qid!r}: box questions need image input; " + NO_IMAGES
    if t not in ("noul", "choice", "score"):
        return f"question {qid!r}: type must be noul, choice or score, got {t!r}"
    c = q.get("criteria")
    if t == "choice" and (not isinstance(c, dict) or len(c) < 2):
        return f"question {qid!r}: choice needs a criteria object with at least 2 options"
    if t == "score" and (not isinstance(c, list) or len(c) < 2):
        return f"question {qid!r}: score needs a criteria list with at least 2 levels"
    if t == "noul" and c is not None and not isinstance(c, dict):
        return f"question {qid!r}: noul criteria must be an object {{true, false}} if given"
    return None


def normalise(q, qid):
    """Fill in what the wire format allows to be missing; canonical JSON for dict/list instructions and option texts."""
    q = dict(q)
    ins = q.get("instructions")
    q["instructions"] = ins if isinstance(ins, str) and ins else (json.dumps(ins, sort_keys=True) if ins else qid)
    if q["type"] == "choice":
        q["criteria"] = {str(k): (v if isinstance(v, str) else json.dumps(v, sort_keys=True)) for k, v in q["criteria"].items()}
    elif q["type"] == "score":
        q["criteria"] = [c if isinstance(c, str) else json.dumps(c, sort_keys=True) for c in q["criteria"]]
    return q


def typed_answer(q, p):
    """Strict typed answer fields from probabilities in options(q) order (noul: [true, false])."""
    p = [float(x) for x in p]
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(p[0], 4)}
    n = len(p)
    if q["type"] == "choice":
        names = list(q["criteria"])
        k = max(range(n), key=lambda i: p[i])
        return {"type": "choice", "choice": names[k], "confidence": round(max(0.0, (n * p[k] - 1) / (n - 1)), 4),
                "probabilities": {nm: round(x, 4) for nm, x in zip(names, p)}}
    score = sum(i * x for i, x in enumerate(p))                      # score = expectation over levels
    med = next(i for i in range(n) if sum(p[: i + 1]) >= 0.5)
    return {"type": "score", "score": round(score, 4), "confidence": round(p[min(n - 1, int(round(score)))], 4),
            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
            "probabilities": {str(i): round(x, 4) for i, x in enumerate(p)}, "_median": med}




class ServiceError(Exception):
    """An HTTP-style error: status (400 / 404 / 413) and detail."""
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class Service:
    """st: a horopter.runtime.Student with its adapter bank; fast: an optional horopter.fast.FastPath on st."""

    def __init__(self, st, prefix="horopter", aliases=("mbnli-3-beta",), release_date="", fast=None):
        from . import families as F
        from . import fields as FL
        from .model import REF, SUP
        from .packing import options
        from .runtime import answer_routed
        self._mods = (F, FL, SUP, REF, options, answer_routed)
        self.st, self.fast, self.lock = st, fast, threading.Lock()
        bank = st.bank
        self.has_router = "router" in bank
        self.families = [f for f in F.FAMILIES if f in bank]
        self.release_date = release_date or st.config.get("release_date", "")
        self.version = f"{CONTRACT} trunk={st.name}" + (f":{st.run}" if st.run else "") + (" fast" if fast else "")
        self.default_model = prefix if self.has_router else f"{prefix}-base"
        self.field_adapter = "extract" if "extract" in bank else None
        served = {prefix: "route"} if self.has_router else {}
        for a in bank:
            if a not in ("router", "evidence", "extract") and not a.startswith("ev_"):   # helpers: never answered from
                served[f"{prefix}-{a}"] = a
        served.update({k + "-strict": v for k, v in list(served.items())})
        for alias in aliases or ():   # the same models under other names (e.g. the beta's mbnli-3-beta)
            if alias and alias != prefix:
                served.update({alias + k[len(prefix):]: v for k, v in list(served.items()) if k.startswith(prefix)})
        self.served = served

    def health(self):
        from . import __version__
        st, fast = self.st, self.fast
        return {"ok": True, "trunk": st.name, "p": st.run, "adapters": sorted(st.bank), "router": self.has_router,
                "device": str(st.device), "fast": dict(fast.cfg) if fast else None, "version": __version__}

    def models(self):
        out, meta = [], self.st.meta
        for name, a in sorted(self.served.items()):
            desc = ("router over " + ", ".join(self.families)) if a == "route" else (
                f"adapter {a}: " + (meta[a].get("kind", "") + (f", dev acc {meta[a].get('dev_acc')}" if a != "base" else "")))
            out.append({"name": name, "description": desc + (" (strict typed body)" if name.endswith("-strict") else ""),
                        "release_date": self.release_date})
        return {"models": out}

    def decide(self, req, clairic=False):
        """req: a request dict (or horopter.api.DecideRequest) -> the response dict every route returns.
        clairic: Clairic route semantics (truncate long states and say so; the state must be non-empty)."""
        from pydantic import ValidationError
        if not isinstance(req, DecideRequest):
            try:
                req = DecideRequest.model_validate(req)
            except ValidationError as e:
                raise ServiceError(400, json.loads(e.json(include_url=False))) from None
        st, served, fast, version = self.st, self.served, self.fast, self.version
        default_model, field_adapter, bank = self.default_model, self.field_adapter, self.st.bank
        F, FL, SUP, REF, options, answer_routed = self._mods
        asked = req.model
        model = default_model if (asked is None or asked in CLAIRIC_NAMES) else asked
        if model not in served:
            raise ServiceError(404, f"this server does not serve {model!r}; it serves {sorted(served)}")
        qs_in = {k: v.model_dump(exclude_none=True) for k, v in (req.questions or {}).items()}
        fs_in = {k: v.model_dump(exclude_none=True) for k, v in (req.fields or {}).items()}
        if not qs_in and not fs_in:
            raise ServiceError(400, "no questions or fields")
        clash = sorted(set(qs_in) & set(fs_in))
        if clash:
            raise ServiceError(400, f"question ids and field names share `answers`; both use {clash}")
        if req.assets or req.images:
            raise ServiceError(400, NO_IMAGES)
        for qid, q in qs_in.items():
            err = check_question(qid, q)
            if err:
                raise ServiceError(400, err)
        for name, spec in fs_in.items():
            err = FL.check(name, spec)
            if err:
                raise ServiceError(400, err)
        if fs_in and not isinstance(req.state, str):
            raise ServiceError(400, "fields need a text state: their spans are offsets into it")
        if clairic and not req.state:
            raise ServiceError(400, "state must be non-empty")
        if req.reference_date:
            try:
                __import__("datetime").date.fromisoformat(req.reference_date)
            except ValueError:
                raise ServiceError(400, "reference_date must be an ISO date (YYYY-MM-DD)") from None
        strict = model.endswith("-strict")
        ns = "mbnli" if model.startswith("mbnli") else "horopter"
        ext = req.horopter or req.mbnli
        opts = ext.model_dump(exclude_none=True) if ext else {}
        ev_mode = (opts.get("evidence") or {}).get("mode", "off")
        if ev_mode not in ("off", "chosen", "all"):
            raise ServiceError(400, "evidence.mode must be off, chosen or all")
        ev_thr = (opts.get("evidence") or {}).get("threshold", 0.5)     # per-token channel probability
        if not isinstance(ev_thr, (int, float)) or not 0 < ev_thr < 1:
            raise ServiceError(400, "evidence.threshold must be a number in (0, 1)")
        debug = bool(opts.get("debug"))
        allow = next((x for x in ((req.options.allow_truncation if req.options else None), opts.get("allow_truncation"))
                      if x is not None), clairic)
        state = render_state(req.state)
        n_state = len(st.tok(state, add_special_tokens=False)["input_ids"])
        limit = st.col.ms - 2
        truncated = n_state > limit
        if truncated and not allow:
            raise ServiceError(413, f"state is {n_state} tokens; this model reads {limit}. A state is never "
                                     f"truncated silently: shorten it or send options.allow_truncation=true")
        qs = {qid: normalise(q, qid) for qid, q in qs_in.items()}
        hidden = {_FIELD + name: FL.field_question(name, spec) for name, spec in fs_in.items()}
        allq = {**qs, **hidden}
        evidence = ev_mode != "off" and isinstance(req.state, str) and not strict
        ev_arg = "all" if hidden else (ev_mode if evidence else False)   # fields read the "yes" option's spans

        def step():                                # routing + answering: the model's part of the request
            route = {}
            routed = list(qs)                      # fields skip the router: it re-reads the state per question, and
            # the adapter hardly changes field answers (all six family adapters tie on the E43-style probes)
            if served[model] == "route" and routed:
                st.use("router")
                rp = st.answer_one_each([(F.router_state(state, allq[q]), F.ROUTER_Q) for q in routed])
                for qid, r in zip(routed, rp):
                    probs = {f: float(x) for f, x in zip(F.FAMILIES, r)}
                    best = max((f for f in probs if f in bank), key=probs.get)
                    route[qid] = {"adapter": best, "p": round(probs[best], 4),
                                  "router": {f: round(x, 4) for f, x in probs.items()}}
            for qid in allq:
                if qid not in route:
                    ad = served[model]
                    if qid in hidden:
                        ad = field_adapter or (ad if ad != "route" else ("classification" if "classification" in bank
                                                                         else "base"))
                    route[qid] = {"adapter": ad, "p": 1.0, "router": None}
                if qid in hidden:
                    route[qid]["own_spans"] = True
            if fast is not None:
                return route, fast.answer(state, allq, route, ev_arg)
            return route, answer_routed(st, state, allq, route, ev_arg)

        t_req = time.perf_counter()
        with self.lock:
            wait_ms = (time.perf_counter() - t_req) * 1000
            route, res = fast.call(step) if fast is not None else step()
        server_ms = (time.perf_counter() - t_req) * 1000

        def n_tok(text):
            return len(st.tok(text, add_special_tokens=False)["input_ids"])

        answers, n_schema = {}, 0
        for qid, q in qs.items():
            d = res[qid]
            a = typed_answer(q, d["p"])
            med = a.pop("_median", None)
            n_schema += n_tok(q["instructions"])
            if not strict:
                names = options(q)
                p = [float(x) for x in d["p"]]
                x = {"relation": {nm: {"support": round(float(r[0]), 4), "refute": round(float(r[1]), 4),
                                       "undetermined": round(float(r[2]), 4)} for nm, r in zip(names, d["rel"])},
                     "abstain": st.abstain(d)}
                x["relation"]["relevance"] = round(1 - d["unrel"], 4)
                if med is not None:
                    x["score"] = {"median": med}
                if evidence:
                    k_best = max(range(len(p)), key=lambda i: p[i])
                    which = range(len(p)) if ev_mode == "all" else [k_best]
                    spans = []
                    for k in which:
                        if d["ev"][k] is None:
                            continue
                        offs, es = d["ev"][k]
                        # both channels, independently: a state can hold words that make an option true AND words
                        # that rule it out (mixed evidence); the role is the channel that fired
                        for role, ch in (("support", SUP), ("refute", REF)):
                            for sp in st.spans(state, offs, es, ch, thr=ev_thr):
                                spans.append(dict(sp, side="state", role=role, option=names[k]))
                    x["evidence"] = spans
                if debug:
                    x["route"] = route[qid]
                a[ns] = x
            answers[qid] = a
        for name, spec in fs_in.items():
            hq = _FIELD + name
            d = res[hq]
            n_schema += n_tok(hidden[hq]["instructions"])
            offs, es = d["ev"][0]                  # option "true": "the text gives it"
            a = FL.answer(name, spec, state, offs, es[:, SUP], d["p"][0],
                          peak_p_null=route[hq]["adapter"] == "extract", reference_date=req.reference_date)
            if debug and not strict:
                a[ns] = {"question": hidden[hq]["instructions"], "p_yes": round(float(d["p"][0]), 4),
                         "route": route[hq], "abstain": st.abstain(d)}
            answers[name] = a
        usage = {"input_tokens": int(n_state + n_schema), "output_tokens": 0}
        body = {"model": asked or model, "usage": usage, "answers": answers}
        if strict:
            return body
        usage.update({"state_tokens": int(min(n_state, limit)), "image_tokens": 0, "schema_tokens": int(n_schema),
                      "passes": 1})
        body["truncated"] = bool(truncated)
        if req.options and req.options.return_state_text:
            body["state_text"] = state if not truncated else state[: max((o[1] for o in st.state_offsets(state)
                                                                          if o), default=0)]
        body[ns] = {"version": version, "state_id": "sha256:" + hashlib.sha256(state.encode()).hexdigest(),
                    "server_ms": round(server_ms, 1), "queue_ms": round(wait_ms, 1),
                    "adapters_used": sorted({r["adapter"] for r in route.values()})}
        return body
