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
    <prefix>                    the default: routed (the router picks a target adapter per question) or, with
                                --serve, one adapter answering directly
    <prefix>-<adapter>          force one adapter: base (the trunk alone) or any loaded non-helper adapter
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

Adapters and routing (docs/ADAPTERS.md). Without these flags: the trunk alone (its own heads and abstention head),
and no adapter is downloaded. Each flag below downloads only the adapters it names, on first use (then cached):
    --adapters REF[,REF...]     load adapters; repeatable. REF: a bundled name (the model's adapters/NAME), a
                                directory, "user/repo[@rev]" (adapter at the repo root) or "user/repo[@rev]:NAME"
                                (adapters/NAME in a repo laid out like a model repo); "all" = every bundled adapter
    --adapter [NAME=]REF        one adapter under a name of your choice; repeatable
    --serve REF                 no routing: <prefix> answers with this adapter ("base" = the trunk alone, the default)
    --router REF                route with this adapter ("base" = zero-shot routing with the trunk); without --route,
                                over the targets its adapter.json lists (the bundled router: the bundled families)
    --route REF [DESCRIPTION]   a routing target (the router's option) and its description; repeatable

    # the bundled helpers: evidence spans and field extraction
    python -m horopter.server --model jmccardle/horopter-0.4B --adapters evidence,extract
    # the bundled router and the six family adapters it routes between (the worked example)
    python -m horopter.server --model jmccardle/horopter-0.4B --router router
    # one adapter, no routing
    python -m horopter.server --model jmccardle/horopter-0.4B --serve ./refunds
    # your router over your adapters, all from one hub repo
    python -m horopter.server --model jmccardle/horopter-0.4B --router someone/horopter-0.4B-retail:router
    # zero-shot routing between two new adapters
    python -m horopter.server --model jmccardle/horopter-0.4B --router base \
        --route ./refunds "a retail refund or return decision" --route ./triage "which team gets a ticket"
"""
from __future__ import annotations

import argparse
import time

from .api import DecideRequest, DecideResponse, WorkersAIResponse
from .service import (CLAIRIC_NAMES, CONTRACT, NO_IMAGES, Service, ServiceError, check_question,  # noqa: F401
                      normalise, render_state, typed_answer)


def build_app(st, prefix: str = "horopter", aliases=("mbnli-3-beta",), release_date: str = "", fast=None,
              routing=None):
    """st: a horopter.runtime.Student with its adapter bank (horopter.runtime.load). -> FastAPI app.
    fast: a horopter.fast.FastPath on st (same answers up to bf16 kernel rounding; docs/FAST_PATH.md).
    routing: a horopter.routing.Routing (default: the bundled router, if loaded)."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    from . import __version__

    svc = Service(st, prefix, aliases, release_date, fast, routing)
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
    ap.add_argument("--adapters", action="append", default=[], metavar="REF[,REF...]",
                    help="adapters to load: bundled names, directories (or a parent of <name>/ adapter dirs), "
                         "'user/repo[@rev]' or 'user/repo[@rev]:NAME'; 'all' = every bundled adapter; repeatable")
    ap.add_argument("--abstain_head", default=None, help="abstention head JSON (default: the model's abstain_head.json)")
    ap.add_argument("--no_adapters", action="store_true", help=argparse.SUPPRESS)   # no-op: loading no bundled adapters is the default
    ap.add_argument("--adapter", action="append", default=[], metavar="[NAME=]SOURCE",
                    help="load an adapter: a directory, 'user/repo' or 'user/repo:path/in/repo'; repeatable")
    ap.add_argument("--allow_trunk_mismatch", action="store_true",
                    help="load --adapter adapters trained on another trunk (their answers will be wrong)")
    ap.add_argument("--serve", default=None, metavar="NAME", help="no routing: the default model answers with this "
                                                                  "adapter ('base' = the trunk alone)")
    ap.add_argument("--router", default=None, metavar="NAME", help="route with this adapter ('base' = the trunk, "
                                                                   "zero-shot)")
    ap.add_argument("--route", action="append", nargs="+", default=[], metavar=("NAME", "DESCRIPTION"),
                    help="a routing target and its description (default: the adapter's adapter.json description); "
                         "repeatable; default targets: the bundled family adapters")
    ap.add_argument("--route_instructions", default=None, help="the router question's instructions")
    ap.add_argument("--route_layout", default=None, choices=["question", "state"],
                    help="what the router reads: the shared state + the question (question) or a fresh state "
                         "built from the question and the start of the state (state, the bundled router's layout)")
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
    refs = [r.strip() for x in a.adapters for r in x.split(",") if r.strip()]
    st = runtime.load(a.model, device=a.device, adapters="all" in refs, max_state=a.max_state, quant=a.quant)
    check = not a.allow_trunk_mismatch
    from pathlib import Path
    for ref in refs:
        if ref == "all":
            continue
        p = Path(ref)
        if p.is_dir() and not (p / "adapter.safetensors").exists():   # a parent of <name>/ adapter dirs
            subs = [d.parent for d in sorted(p.glob("*/adapter.safetensors")) if d.parent.name != "smoke"]
            if not subs:
                ap.error(f"--adapters {ref}: no adapter.safetensors in it or in its subdirectories")
            for d in subs:
                st.add_adapter_from(d, check_trunk=check)
        else:
            st.add_adapter_from(ref, check_trunk=check)
    for spec in a.adapter:
        name, eq, src = spec.partition("=")
        if not eq or "/" in name:                  # no NAME= (a path may contain "=")
            name, src = None, spec
        st.add_adapter_from(src, name, check_trunk=check)
    from . import routing as RT
    if any(len(r) > 2 for r in a.route):
        ap.error("--route takes NAME and an optional DESCRIPTION (quote a description with spaces)")
    try:
        serve, router, routes = RT.prepare(st, a.serve, a.router, [(r[0], r[1] if len(r) > 1 else None)
                                                                   for r in a.route] or None, check_trunk=check)
        rt = RT.configure(st, serve=serve, router=router, routes=routes, instructions=a.route_instructions,
                          layout=a.route_layout)
    except (RT.RoutingError, FileNotFoundError, ValueError) as e:
        ap.error(str(e))
    if a.abstain_head:
        st.abstain_head = A.AbstainHead.from_json(a.abstain_head)
    print(f"loaded {st.name} + adapters {sorted(st.bank)} in {time.perf_counter() - t0:.1f}s; abstain head "
          f"{st.abstain_head.name if st.abstain_head else 'placeholder'}; routing {rt.info()}", flush=True)
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
    uvicorn.run(build_app(st, a.prefix, [x for x in a.alias.split(",") if x], fast=fp, routing=rt), host=a.host,
                port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
