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
import time

from .api import DecideRequest, DecideResponse, WorkersAIResponse
from .service import (CLAIRIC_NAMES, CONTRACT, NO_IMAGES, Service, ServiceError, check_question,  # noqa: F401
                      normalise, render_state, typed_answer)


def build_app(st, prefix: str = "horopter", aliases=("mbnli-3-beta",), release_date: str = "", fast=None):
    """st: a horopter.runtime.Student with its adapter bank (horopter.runtime.load). -> FastAPI app.
    fast: a horopter.fast.FastPath on st (same answers up to bf16 kernel rounding; docs/FAST_PATH.md)."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    from . import __version__

    svc = Service(st, prefix, aliases, release_date, fast)
    app = FastAPI(title="Horopter", version=__version__,
                  description="Typed decisions (Jev System One questions) and typed extraction (Clairic fields) over "
                              "one cached state, with relation, abstention and evidence spans.")
    app.state.service = svc

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):   # 400, as Jev and Clairic answer
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})

    def decide(req, clairic):
        try:
            return svc.decide(req, clairic=clairic)
        except ServiceError as e:
            raise HTTPException(e.status, e.detail) from None

    app.get("/healthz")(svc.health)
    app.get("/v1/models")(svc.models)
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
