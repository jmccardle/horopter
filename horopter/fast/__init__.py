# SPDX-License-Identifier: MIT
"""Fast path: the same answers as horopter.runtime.Student up to bf16 kernel rounding, much faster.

    from horopter import runtime, fast
    st = runtime.load("path/to/horopter-0.4B")
    fp = fast.FastPath(st)                         # CUDA: graphs + compile + Flex state pass + multi-adapter pass
    res = fp.call(lambda: fp.answer(state, qs, route, evidence))   # every GPU call goes through fp.call
    python -m horopter.server --model path/to/horopter-0.4B --fast

What it does (docs/FAST_PATH.md has the measurements):
    state cache    the state encoding is question-blind and adapter-blind, so a request splits into a state pass
                   (cached by state token ids: an LRU of `slots` states) and a question pass that reads the cached
                   K/V. A state asked about again costs only the question tokens.            (stateq, graphq)
    flat bank      every adapter's tensors live in one buffer; an adapter swap is one copy per dtype.      (bank)
    CUDA graphs    both passes replay as graphs (one per padded shape), so the CPU no longer launches kernels
                   one by one.                                                                      (graphq)
    Flex           block-sparse attention for the state pass's sliding-window layers: pays from ~500 tokens.
                                                                                                  (flexattn)
    multi-adapter  every adapter of a request in ONE question pass (segmented LoRA), heads and answer maths in
                   the same graph, one device-to-host copy.                                 (multilora, vheads)
On CUDA the trunk runs with bf16 weights (the answer head stays fp32), as the fast path was measured; on CPU it stays
fp32 and only the state cache (+ torch.compile) is used, because a CPU is compute-bound: graphs remove nothing and
the multi-adapter pass's padding costs real time there.

CUDA graphs are captured and replayed on one worker thread that this object owns; call every model operation through
`call` (the server does). A graph is captured the first time a shape is seen (padded state length, question tokens,
adapters per request); pass `precapture` (multilora.default_shapes(), or the output of `discover`) to pay that at
start-up instead. Captured graphs and the state slots stay resident: on small shared-memory devices precapture only
the shapes the deployment uses.
"""
import queue
import threading
from concurrent.futures import Future

import torch

__all__ = ["FastPath", "to_bf16"]


@torch.no_grad()
def to_bf16(st):
    """Trunk and head parameters (and the adapter bank) to bf16; the answer head and every buffer (rotary inv_freq)
    keep their dtype. Student itself keeps fp32 weights under bf16 autocast."""
    for p in st.model.parameters():
        p.data = p.data.to(torch.bfloat16)
    st.model.float = lambda *a, **k: st.model       # nothing may upcast it again
    st.ans.float()
    conv = lambda sd: {k: v.to(torch.bfloat16) if not k.startswith("answer.") else v.float() for k, v in sd.items()}
    st.bank = {name: conv(sd) for name, sd in st.bank.items()}
    st.base_state = st.bank["base"]


class FastPath:
    def __init__(self, st, graphs=None, compile=True, flex=None, multilora=None, bf16=None, slots=3, cache_size=4,
                 precapture=None):
        cuda = torch.device(st.device).type == "cuda"
        self.st = st
        self.cfg = dict(graphs=cuda if graphs is None else graphs, compile=compile, flex=cuda if flex is None else flex,
                        multilora=cuda if multilora is None else multilora, bf16=cuda if bf16 is None else bf16,
                        slots=slots, cache_size=cache_size)
        if self.cfg["graphs"] and not cuda:
            raise ValueError("CUDA graphs need a CUDA device")
        if self.cfg["flex"] and not self.cfg["graphs"]:
            raise ValueError("flex runs inside the graphed state pass (graphs=True)")
        self.ms, self.captured = None, 0
        self._q = queue.Queue()
        ready = Future()
        self._th = threading.Thread(target=self._run, args=(ready, precapture), daemon=True, name="horopter-fast")
        self._th.start()
        ready.result()                              # raises if the set-up failed

    # ------------------------------------------------------------------ worker
    def _setup(self, precapture):
        from . import bank
        c, st = self.cfg, self.st
        if c["bf16"]:
            to_bf16(st)
        if c["compile"]:
            for lay in st.model.bert.layers:        # the full forward (router batches with several states)
                lay.compile(dynamic=True)
        self.split, self.fb = bank.install(st, compile=c["compile"], graphs=c["graphs"], slots=c["slots"],
                                           cache_size=c["cache_size"], flex=c["flex"])
        if c["multilora"]:
            from . import multilora
            self.ms = multilora.MultiSplit(st, self.split if c["graphs"] else None)
            if precapture:
                self.captured = self.ms.precapture(precapture)

    def _run(self, ready, precapture):
        try:
            with torch.no_grad():
                self._setup(precapture)
        except BaseException as e:   # noqa: BLE001
            ready.set_exception(e)
            return
        ready.set_result(True)
        while True:
            job = self._q.get()
            if job is None:
                return
            fn, fut = job
            try:
                with torch.no_grad():
                    fut.set_result(fn())
            except BaseException as e:   # noqa: BLE001
                fut.set_exception(e)

    def call(self, fn):
        """Run fn() on the worker thread (the one that owns the graphs) and return its result."""
        if threading.current_thread() is self._th:
            return fn()
        fut = Future()
        self._q.put((fn, fut))
        return fut.result()

    def close(self):
        self._q.put(None)
        self._th.join()

    # ------------------------------------------------------------------ answering (call these through `call`)
    def answer(self, state, qs, route, evidence):
        """The server's answer step after routing: route {qid: {"adapter": name}} -> {qid: Student.answer dict}.
        Spans come from the "evidence" adapter when one is loaded (its answers are not used), as in the server."""
        if self.ms is not None:
            from .multilora import serve_request
            return serve_request(self.ms, state, qs, route, evidence)
        st, res = self.st, {}
        ev_ad = "evidence" in st.bank and evidence
        for ad in sorted({r["adapter"] for r in route.values()}):
            st.use(ad)
            res.update(st.answer(state, {q: qs[q] for q in qs if route[q]["adapter"] == ad},
                                 evidence=bool(evidence) and not ev_ad))
        if ev_ad:
            st.use("evidence")
            for qid, d in st.answer(state, qs, evidence=True).items():
                res[qid]["ev"] = d["ev"]
        return res

    def put_state(self, state):
        """Encode a (rendered) state into the cache now, so a later question about it costs only the question pass."""
        st = self.st
        ids = st.col._state(state)[0]
        t = torch.tensor([ids], device=st.device)
        return self.split.encode_state(t, tuple(ids))

    def discover(self, requests):
        """Graph shapes a list of requests [(state, qs, route, evidence)] needs (multi-adapter path only), for
        `precapture` on the next start. Runs each request's state pass; captures nothing."""
        return self.ms.discover(requests) if self.ms is not None else []
