# SPDX-License-Identifier: MIT
"""HTTP server: one handler for Jev-style typed questions and Clairic-style typed fields over one cached state.

    POST /v1/systemone   Jev System One route  {"model", "state", "questions": {id: {type, instructions, criteria}}}
    POST /v1/clairic     Clairic route         {"model": "clairic", "state", "fields": {name: {type, ...}}}
    POST /ai/run/@cf/cloudflare/clairic, /client/v4/accounts/{account}/ai/run/@cf/cloudflare/clairic
                         Workers AI REST paths: the same answer wrapped as {"result", "success", "errors", "messages"}
    POST /v1/horopter    our own name for the same handler (questions and fields together)
    GET  /v1/models, /healthz, /openapi.json, /docs

Every route takes the same request (horopter.api.DecideRequest: questions, fields, or both) and returns the same body,
a superset of the Jev answer (model, usage, answers) and the Clairic answer (model, answers, truncated, usage,
state_text). Question ids and field names share `answers`. Images (Jev `assets`, Clairic `images`) and Jev's `box`
question are accepted by the schema and refused with a 400: this model has no image input yet.

Models (``model`` field; a name this server does not serve is a 404, never a silent substitution; ``clairic`` and
``@cf/cloudflare/clairic`` map to the default model and are echoed back):
    <prefix>                    the router picks an adapter per question (only if a "router" adapter is loaded)
    <prefix>-<adapter>          force one adapter: base (the trunk alone) or a family / helper-trained adapter
    any of the above + "-strict"  strict Jev body: no extension blocks, no Clairic keys
The prefix defaults to "horopter"; every name is also served under each --alias prefix (default "mbnli-3-beta", the
beta's name). Extension options may be sent as ``horopter`` or ``mbnli``; the extension blocks in the response are
named ``mbnli`` for mbnli-* model names and ``horopter`` otherwise.

Question answers (Jev): noul -> noul (P(true)); choice -> choice, confidence, probabilities; score -> score (expected
level), confidence, legend, probabilities. Extension block per answer:
    relation[option]  the trunk's stance softmax for that option block (support / refute / undetermined); under the
                      hard bottleneck ``probabilities`` is the normalised support, so they agree by construction
    relation.relevance  1 - p(unrelated) from the question block
    abstain           with an abstention head: model_uncertainty = the head's P(wrong) from 21 answer / relation /
                      evidence scalars, reasons.unrelated = p(unrelated), p = max of the two. Without one, a
                      placeholder: model_uncertainty = normalised entropy
    evidence          per option, both state channels independently: spans whose SUPPORT channel (words that make the
                      option true) or CONFLICT channel (words that rule it out, served as "refute") scores >=
                      evidence.threshold, merged, top 3 per channel, snapped to whole words; from the "evidence"
                      adapter when one is loaded; string states only
    score.median      score questions: the median level

Field answers (Clairic): value, guess, p_null, confidence, span (or spans for lists); see horopter.fields. Each field
is a hidden yes/no question ("Does the text give ...?") answered with evidence from its own adapter (the "extract"
adapter when one is loaded, else the routed family adapter).

    python -m horopter.server --model jmccardle/horopter-0.4B [--host 127.0.0.1] [--port 8420] [--device cpu] [--fast]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time

from .api import DecideRequest, DecideResponse, WorkersAIResponse

CONTRACT = "3.1.0+contract2"
_LOCK = threading.Lock()
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


def build_app(st, prefix: str = "horopter", aliases=("mbnli-3-beta",), release_date: str = "", fast=None):
    """st: a horopter.runtime.Student with its adapter bank (horopter.runtime.load). -> FastAPI app.
    fast: a horopter.fast.FastPath on st (same answers up to bf16 kernel rounding; docs/FAST_PATH.md)."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    from . import __version__
    from . import families as F
    from . import fields as FL
    from .model import REF, SUP
    from .packing import options
    from .runtime import answer_routed

    bank, meta = st.bank, st.meta
    has_router = "router" in bank
    families = [f for f in F.FAMILIES if f in bank]
    release_date = release_date or st.config.get("release_date", "")
    version = f"{CONTRACT} trunk={st.name}" + (f":{st.run}" if st.run else "") + (" fast" if fast else "")
    default_model = prefix if has_router else f"{prefix}-base"
    field_adapter = "extract" if "extract" in bank else None

    served = {prefix: "route"} if has_router else {}
    for a in bank:
        if a not in ("router", "evidence", "extract") and not a.startswith("ev_"):   # helpers: never answered from
            served[f"{prefix}-{a}"] = a
    served.update({k + "-strict": v for k, v in list(served.items())})
    for alias in aliases or ():   # the same models under other names (e.g. the beta's mbnli-3-beta)
        if alias and alias != prefix:
            served.update({alias + k[len(prefix):]: v for k, v in list(served.items()) if k.startswith(prefix)})

    app = FastAPI(title="Horopter", version=__version__,
                  description="Typed decisions (Jev System One questions) and typed extraction (Clairic fields) over "
                              "one cached state, with relation, abstention and evidence spans.")

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):   # 400, as Jev and Clairic answer
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "trunk": st.name, "p": st.run, "adapters": sorted(bank), "router": has_router,
                "device": str(st.device), "fast": dict(fast.cfg) if fast else None, "version": __version__}

    @app.get("/v1/models")
    def models():
        out = []
        for name, a in sorted(served.items()):
            desc = ("router over " + ", ".join(families)) if a == "route" else (
                f"adapter {a}: " + (meta[a].get("kind", "") + (f", dev acc {meta[a].get('dev_acc')}" if a != "base" else "")))
            out.append({"name": name, "description": desc + (" (strict typed body)" if name.endswith("-strict") else ""),
                        "release_date": release_date})
        return {"models": out}

    def decide(req: DecideRequest, clairic: bool) -> dict:
        asked = req.model
        model = default_model if (asked is None or asked in CLAIRIC_NAMES) else asked
        if model not in served:
            raise HTTPException(404, f"this server does not serve {model!r}; it serves {sorted(served)}")
        qs_in = {k: v.model_dump(exclude_none=True) for k, v in (req.questions or {}).items()}
        fs_in = {k: v.model_dump(exclude_none=True) for k, v in (req.fields or {}).items()}
        if not qs_in and not fs_in:
            raise HTTPException(400, "no questions or fields")
        clash = sorted(set(qs_in) & set(fs_in))
        if clash:
            raise HTTPException(400, f"question ids and field names share `answers`; both use {clash}")
        if req.assets or req.images:
            raise HTTPException(400, NO_IMAGES)
        for qid, q in qs_in.items():
            err = check_question(qid, q)
            if err:
                raise HTTPException(400, err)
        for name, spec in fs_in.items():
            err = FL.check(name, spec)
            if err:
                raise HTTPException(400, err)
        if fs_in and not isinstance(req.state, str):
            raise HTTPException(400, "fields need a text state: their spans are offsets into it")
        if clairic and not req.state:
            raise HTTPException(400, "state must be non-empty")
        if req.reference_date:
            try:
                __import__("datetime").date.fromisoformat(req.reference_date)
            except ValueError:
                raise HTTPException(400, "reference_date must be an ISO date (YYYY-MM-DD)") from None
        strict = model.endswith("-strict")
        ns = "mbnli" if model.startswith("mbnli") else "horopter"
        ext = req.horopter or req.mbnli
        opts = ext.model_dump(exclude_none=True) if ext else {}
        ev_mode = (opts.get("evidence") or {}).get("mode", "off")
        if ev_mode not in ("off", "chosen", "all"):
            raise HTTPException(400, "evidence.mode must be off, chosen or all")
        ev_thr = (opts.get("evidence") or {}).get("threshold", 0.5)     # per-token channel probability
        if not isinstance(ev_thr, (int, float)) or not 0 < ev_thr < 1:
            raise HTTPException(400, "evidence.threshold must be a number in (0, 1)")
        debug = bool(opts.get("debug"))
        allow = next((x for x in ((req.options.allow_truncation if req.options else None), opts.get("allow_truncation"))
                      if x is not None), clairic)
        state = render_state(req.state)
        n_state = len(st.tok(state, add_special_tokens=False)["input_ids"])
        limit = st.col.ms - 2
        truncated = n_state > limit
        if truncated and not allow:
            raise HTTPException(413, f"state is {n_state} tokens; this model reads {limit}. A state is never "
                                     f"truncated silently: shorten it or send options.allow_truncation=true")
        qs = {qid: normalise(q, qid) for qid, q in qs_in.items()}
        hidden = {_FIELD + name: FL.field_question(name, spec) for name, spec in fs_in.items()}
        allq = {**qs, **hidden}
        evidence = ev_mode != "off" and isinstance(req.state, str) and not strict
        ev_arg = "all" if hidden else (ev_mode if evidence else False)   # fields read the "yes" option's spans

        def step():                                # routing + answering: the model's part of the request
            route = {}
            routed = [q for q in allq if q in qs or field_adapter is None]
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
                    ad = field_adapter if (qid in hidden and field_adapter) else served[model]
                    route[qid] = {"adapter": ad, "p": 1.0, "router": None}
                if qid in hidden:
                    route[qid]["own_spans"] = True
            if fast is not None:
                return route, fast.answer(state, allq, route, ev_arg)
            return route, answer_routed(st, state, allq, route, ev_arg)

        t_req = time.perf_counter()
        with _LOCK:
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

    ok = {200: {"model": DecideResponse}}

    @app.post("/v1/systemone", responses=ok, summary="Jev System One route (questions; fields accepted too)")
    def systemone(req: DecideRequest):
        return decide(req, clairic=False)

    @app.post("/v1/horopter", responses=ok, summary="Questions and fields over one state")
    def horopter_route(req: DecideRequest):
        return decide(req, clairic=False)

    @app.post("/v1/clairic", responses=ok, summary="Clairic route (fields; questions accepted too)")
    def clairic_route(req: DecideRequest):
        return decide(req, clairic=True)

    def workers(req: DecideRequest):
        return {"result": decide(req, clairic=True), "success": True, "errors": [], "messages": []}

    app.post("/ai/run/@cf/cloudflare/clairic", responses={200: {"model": WorkersAIResponse}},
             summary="Workers AI REST shape of the Clairic route")(workers)
    app.post("/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clairic",
             responses={200: {"model": WorkersAIResponse}}, include_in_schema=False)(workers)
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="an exported model directory (scripts/export_hf.py) or hub repo id")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"], help="default: cuda if available")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads for --device cpu")
    ap.add_argument("--max_state", type=int, default=None, help="state window in tokens (default: the model's)")
    ap.add_argument("--prefix", default="horopter", help="served model id prefix")
    ap.add_argument("--alias", default="mbnli-3-beta", help="comma list of extra prefixes serving the same models "
                                                            "('' = none)")
    ap.add_argument("--adapters", default=None, help="extra adapter dir (<name>/adapter.safetensors), added to the "
                                                     "model's own adapters/")
    ap.add_argument("--abstain_head", default=None, help="abstention head JSON (default: the model's abstain_head.json)")
    ap.add_argument("--no_adapters", action="store_true", help="serve the trunk alone")
    ap.add_argument("--quant", default=None, choices=["w8", "w4g32", "w4g128"],
                    help="load quantized/model-<quant>.safetensors (Q8 / Q4 trunk) instead of model.safetensors")
    ap.add_argument("--fast", action="store_true", help="horopter.fast: state cache + CUDA graphs + Flex + one pass for "
                                                         "all adapters (cuda; bf16 weights); state cache only on cpu")
    ap.add_argument("--slots", type=int, default=3, help="--fast: states kept in the cache")
    ap.add_argument("--precapture", default="none", choices=["none", "default"],
                    help="--fast on cuda: capture the default graph shapes at start-up (first requests then pay no "
                         "capture; costs start-up time and GPU memory)")
    a = ap.parse_args()
    import torch
    import uvicorn

    from . import abstain as A
    from . import runtime
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    st = runtime.load(a.model, device=a.device, adapters=not a.no_adapters, max_state=a.max_state, quant=a.quant)
    if a.adapters:
        from pathlib import Path
        for d in sorted(Path(a.adapters).glob("*/adapter.safetensors")):
            if d.parent.name != "smoke":
                st.add_adapter(d.parent.name, d.parent)
    if a.abstain_head:
        st.abstain_head = A.AbstainHead.from_json(a.abstain_head)
    print(f"loaded {st.name} + adapters {sorted(st.bank)} in {time.perf_counter() - t0:.1f}s; abstain head "
          f"{st.abstain_head.name if st.abstain_head else 'placeholder'}", flush=True)
    fp = None
    if a.fast:
        from . import fast
        t0 = time.perf_counter()
        pre = None
        if a.precapture == "default" and torch.device(st.device).type == "cuda":
            from .fast.multilora import default_shapes
            pre = default_shapes()
        fp = fast.FastPath(st, slots=a.slots, precapture=pre)
        print(f"fast path {fp.cfg} ready in {time.perf_counter() - t0:.1f}s ({fp.captured} graphs precaptured)",
              flush=True)
    uvicorn.run(build_app(st, a.prefix, [x for x in a.alias.split(",") if x], fast=fp), host=a.host, port=a.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
