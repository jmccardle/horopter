# SPDX-License-Identifier: MIT
"""Fast path for a horopter.runtime.Student, behaviour-preserving (same answers up to kernel rounding):
    1. state KV cache (stateq): one state encoding per state, shared by every adapter pass and by later requests
    2. flat adapter bank: every adapter's 255 tensors live in one buffer per dtype, and the live parameters are views
       of an "active" buffer, so a swap is one copy kernel per dtype instead of 255
    3. (compile) the split layer functions compiled with dynamic shapes

    from horopter.fast import bank; split, fb = bank.install(st, compile=True)     # usually via horopter.fast.FastPath
"""
import torch

from . import stateq


class FlatBank:
    def __init__(self, st):
        self.st = st
        live = {}
        for n, mod in st.model.bert.named_modules():
            if isinstance(mod, stateq.LoRALinear):
                live[f"bert.{n}.A"], live[f"bert.{n}.B"] = (mod, "A"), (mod, "B")
        for n, p in st.model.named_parameters():
            if not n.startswith("bert."):
                live[n] = (st.model.get_submodule(n.rsplit(".", 1)[0]) if "." in n else st.model, n.rsplit(".", 1)[-1])
        for n, _ in st.ans.named_parameters():
            live[f"answer.{n}"] = (st.ans.get_submodule(n.rsplit(".", 1)[0]) if "." in n else st.ans, n.rsplit(".", 1)[-1])
        self.keys = sorted(live)
        cur = {k: getattr(*live[k]) for k in self.keys}
        self.dtypes = sorted({cur[k].dtype for k in self.keys}, key=str)
        self.layout = {}                          # key -> (dtype, offset, numel, shape)
        size = dict.fromkeys(self.dtypes, 0)
        for k in self.keys:
            t = cur[k]
            self.layout[k] = (t.dtype, size[t.dtype], t.numel(), t.shape)
            size[t.dtype] += t.numel()
        dev = cur[self.keys[0]].device
        self.active = {dt: torch.empty(size[dt], dtype=dt, device=dev) for dt in self.dtypes}
        self.flat = {name: self.pack(sd) for name, sd in st.bank.items()}
        for k in self.keys:                       # live parameters become views of the active buffer
            dt, o, n, shape = self.layout[k]
            mod, attr = live[k]
            v = self.active[dt][o:o + n].view(shape)
            v.copy_(cur[k])
            setattr(mod, attr, torch.nn.Parameter(v, requires_grad=False))
        st.load_adapter = self.load

    def pack(self, sd):
        out = {dt: torch.empty_like(a) for dt, a in self.active.items()}
        for k in self.keys:
            dt, o, n, _ = self.layout[k]
            out[dt][o:o + n].copy_(sd[k].reshape(-1).to(dt))
        return out

    @torch.no_grad()
    def load(self, sd, name="?"):
        src = self.flat.get(name)
        if src is None:                           # an adapter added after install
            src = self.flat[name] = self.pack(sd)
        for dt in self.dtypes:
            self.active[dt].copy_(src[dt])
        self.st.loaded = name


def install(st, compile=False, cache_size=4, graphs=False, slots=3, flex=False):
    if compile:
        import torch._dynamo.config as dc
        for k in ("recompile_limit", "cache_size_limit"):   # the split layer functions are called with 28 layers
            if hasattr(dc, k):
                setattr(dc, k, 256)
    fb = FlatBank(st)                    # first: graphs must see the adapter tensors at their final addresses
    if graphs:
        from . import graphq
        return graphq.install(st, slots=slots, compile=compile, flex=flex), fb   # flex: block-sparse state-pass attention
    sp = stateq.install(st, cache_size=cache_size, compile=compile)
    return sp, fb
