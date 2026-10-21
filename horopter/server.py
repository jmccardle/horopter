# SPDX-License-Identifier: MIT
"""HTTP server: ``POST /v1/systemone`` in front of a Horopter model with routed question-path LoRA adapters.

    POST /v1/systemone  {"model": "horopter", "state": "..." | {...}, "questions": {id: {type, instructions, criteria}},
                         "mbnli": {"evidence": {"mode": "off|chosen|all", "threshold": 0.5}, "debug": false,
                                   "allow_truncation": false}}
    GET  /v1/models     what this server serves
    GET  /healthz

Models (``model`` field; a name this server does not serve is a 404, never a silent substitution):
    <prefix>                    the router picks an adapter per question (only if a "router" adapter is loaded)
    <prefix>-<adapter>          force one adapter: base (the trunk alone) or a family / helper-trained adapter
    any of the above + "-strict"  strict typed body: no ``mbnli`` extension blocks
The prefix defaults to "horopter"; every name is also served under each --alias prefix (default "mbnli-3-beta", the
beta's name). The request's extension options may be sent as ``mbnli`` or ``horopter``; the response block is ``mbnli``.

Answer fields (per question): noul -> noul (P(true)); choice -> choice, confidence, probabilities; score -> score
(expected level), confidence, legend, probabilities. Extension block ``mbnli`` per answer:
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

    python -m horopter.server --model path/to/horopter-0.4B [--host 127.0.0.1] [--port 8420] [--device cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from typing import Any, Union

from pydantic import BaseModel

CONTRACT = "3.0.0+contract1"
_LOCK = threading.Lock()


class SystemOneRequest(BaseModel):   # module scope: FastAPI resolves string annotations against module globals
    state: Union[str, dict, list]
    questions: dict[str, dict[str, Any]]
    model: str | None = None
    mbnli: dict[str, Any] | None = None
    horopter: dict[str, Any] | None = None


def render_state(state):
    """One canonical renderer for training and serving: strings as is, JSON sorted and compact."""
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True, separators=(",", ":"))


def check_question(qid, q):
    t = q.get("type")
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


def build_app(st, prefix: str = "horopter", aliases=("mbnli-3-beta",), release_date: str = ""):
    """st: a horopter.runtime.Student with its adapter bank (horopter.runtime.load). -> FastAPI app."""
    from fastapi import FastAPI, HTTPException

    from . import families as F
    from .model import REF, SUP
    from .packing import options

    bank, meta = st.bank, st.meta
    has_router = "router" in bank
    families = [f for f in F.FAMILIES if f in bank]
    release_date = release_date or st.config.get("release_date", "")
    version = f"{CONTRACT} trunk={st.name}" + (f":{st.run}" if st.run else "")

    served = {prefix: "route"} if has_router else {}
    for a in bank:
        if a not in ("router", "evidence") and not a.startswith("ev_"):   # helper adapters: never answered from
            served[f"{prefix}-{a}"] = a
    served.update({k + "-strict": v for k, v in list(served.items())})
    for alias in aliases or ():   # the same models under other names (e.g. the beta's mbnli-3-beta)
        if alias and alias != prefix:
            served.update({alias + k[len(prefix):]: v for k, v in list(served.items()) if k.startswith(prefix)})

    app = FastAPI(title="Horopter")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "trunk": st.name, "p": st.run, "adapters": sorted(bank), "router": has_router,
                "device": str(st.device)}

    @app.get("/v1/models")
    def models():
        out = []
        for name, a in sorted(served.items()):
            desc = ("router over " + ", ".join(families)) if a == "route" else (
                f"adapter {a}: " + (meta[a].get("kind", "") + (f", dev acc {meta[a].get('dev_acc')}" if a != "base" else "")))
            out.append({"name": name, "description": desc + (" (strict typed body)" if name.endswith("-strict") else ""),
                        "release_date": release_date})
        return {"models": out}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        model = req.model or (prefix if has_router else f"{prefix}-base")
        if model not in served:
            raise HTTPException(404, f"this server does not serve {model!r}; it serves {sorted(served)}")
        if not req.questions:
            raise HTTPException(400, "no questions")
        for qid, q in req.questions.items():
            err = check_question(qid, q)
            if err:
                raise HTTPException(400, err)
        strict = model.endswith("-strict")
        opts = req.mbnli or req.horopter or {}
        ev_mode = (opts.get("evidence") or {}).get("mode", "off")
        if ev_mode not in ("off", "chosen", "all"):
            raise HTTPException(400, "mbnli.evidence.mode must be off, chosen or all")
        ev_thr = (opts.get("evidence") or {}).get("threshold", 0.5)     # per-token channel probability
        if not isinstance(ev_thr, (int, float)) or not 0 < ev_thr < 1:
            raise HTTPException(400, "mbnli.evidence.threshold must be a number in (0, 1)")
        debug = bool(opts.get("debug"))
        state = render_state(req.state)
        n_state = len(st.tok(state, add_special_tokens=False)["input_ids"])
        limit = st.col.ms - 2
        if n_state > limit and not opts.get("allow_truncation"):
            raise HTTPException(413, f"state is {n_state} tokens; this model reads {limit}. A state is never "
                                     f"truncated silently: shorten it or send mbnli.allow_truncation=true")
        qs = {qid: normalise(q, qid) for qid, q in req.questions.items()}
        evidence = ev_mode != "off" and isinstance(req.state, str) and not strict

        t_req = time.perf_counter()
        with _LOCK:
            wait_ms = (time.perf_counter() - t_req) * 1000
            route = {}
            if served[model] == "route":
                st.use("router")
                ids = list(qs)                     # one hidden routing question per asked question, batched
                rp = st.answer_one_each([(F.router_state(state, qs[q]), F.ROUTER_Q) for q in ids])
                for qid, r in zip(ids, rp):
                    probs = {f: float(x) for f, x in zip(F.FAMILIES, r)}
                    best = max((f for f in probs if f in bank), key=probs.get)
                    route[qid] = {"adapter": best, "p": round(probs[best], 4),
                                  "router": {f: round(x, 4) for f, x in probs.items()}}
            else:
                for qid in qs:
                    route[qid] = {"adapter": served[model], "p": 1.0, "router": None}
            res = {}
            ev_ad = "evidence" in bank and evidence
            for ad in sorted({r["adapter"] for r in route.values()}):
                st.use(ad)
                res.update(st.answer(state, {q: qs[q] for q in qs if route[q]["adapter"] == ad},
                                     evidence=evidence and not ev_ad))
            if ev_ad:                              # spans from the evidence adapter; answers stay the family's
                st.use("evidence")
                for qid, d in st.answer(state, qs, evidence=True).items():
                    res[qid]["ev"] = d["ev"]
        server_ms = (time.perf_counter() - t_req) * 1000

        answers = {}
        n_q_tok = 0
        for qid, q in qs.items():
            d = res[qid]
            a = typed_answer(q, d["p"])
            med = a.pop("_median", None)
            n_q_tok += len(st.tok(q["instructions"], add_special_tokens=False)["input_ids"])
            if not strict:
                names = options(q)
                p = [float(x) for x in d["p"]]
                ext = {"relation": {nm: {"support": round(float(r[0]), 4), "refute": round(float(r[1]), 4),
                                         "undetermined": round(float(r[2]), 4)} for nm, r in zip(names, d["rel"])},
                       "abstain": st.abstain(d)}
                ext["relation"]["relevance"] = round(1 - d["unrel"], 4)
                if med is not None:
                    ext["score"] = {"median": med}
                if evidence:
                    k_best = max(range(len(p)), key=lambda i: p[i])
                    which = range(len(p)) if ev_mode == "all" else [k_best]
                    spans = []
                    for k in which:
                        offs, es = d["ev"][k]
                        # both channels, independently: a state can hold words that make an option true AND words
                        # that rule it out (mixed evidence); the role is the channel that fired
                        for role, ch in (("support", SUP), ("refute", REF)):
                            for sp in st.spans(state, offs, es, ch, thr=ev_thr):
                                spans.append(dict(sp, side="state", role=role, option=names[k]))
                    ext["evidence"] = spans
                if debug:
                    ext["route"] = route[qid]
                a["mbnli"] = ext
            answers[qid] = a
        body = {"model": model, "usage": {"input_tokens": int(n_state + n_q_tok), "output_tokens": 0}, "answers": answers}
        if not strict:
            body["mbnli"] = {"version": version,
                             "state_id": "sha256:" + hashlib.sha256(state.encode()).hexdigest(),
                             "server_ms": round(server_ms, 1), "queue_ms": round(wait_ms, 1),
                             "adapters_used": sorted({r["adapter"] for r in route.values()})}
        return body

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
    uvicorn.run(build_app(st, a.prefix, [x for x in a.alias.split(",") if x]), host=a.host, port=a.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
