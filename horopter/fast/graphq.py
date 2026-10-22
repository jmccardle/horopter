# SPDX-License-Identifier: MIT
"""The split forward (stateq) with both passes replayed as CUDA graphs.

The state cache lives in `slots` fixed buffers sized for the model's max state (per layer K, V and the final state
hidden states), so every address a graph reads or writes is static:
    state pass      one graph per (slot, S padded to 64): writes the slot's K, V and state hidden states
    question pass   one graph per (slot, S bucket, question tokens padded to 32): reads the slot, writes hq
Pad tokens are masked out of every real row (pads attend only themselves), so answers equal the unpadded split up to
kernel rounding. The adapter tensors are views of the flat bank's active buffer (bank.FlatBank), so one graph serves
every adapter. Heads run eagerly on the graph outputs. Multi-pack batches (the router with several questions) keep the
full forward.

    sp = graphq.install(st, slots=3)    # after bank.FlatBank(st); use bank.install(st, graphs=True)
"""
import collections

import torch

from . import stateq

SB, QB = 64, 32   # pad buckets: state tokens, question tokens


def _ceil(n, b):
    return -(-n // b) * b


class GSplit:
    def __init__(self, st, slots=3, compile=True, flex=False):
        self.st, self.m = st, st.model
        self.full = self.m.forward
        m = self.m
        L = len(m.bert.layers)
        at = m.bert.layers[0].attn
        self.H, self.hd, self.d = at.num_heads, at.head_dim, m.bert.config.hidden_size
        self.Smax = _ceil(st.col.ms, SB)
        dev, dt = next(m.parameters()).device, torch.bfloat16    # cached K, V are autocast (bf16) outputs
        self.dev, self.dt = dev, dt
        self.kv = [torch.zeros(L, 2, 1, self.H, self.Smax, self.hd, device=dev, dtype=dt) for _ in range(slots)]
        self.hs = [torch.zeros(1, self.Smax, self.d, device=dev, dtype=dt) for _ in range(slots)]
        self.slot_of = collections.OrderedDict()    # state key -> (slot, S)
        self.free = list(range(slots))
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.sl, self.ql = stateq.state_layer, stateq.question_layer
        if compile:
            import torch._dynamo.config as dc
            for k in ("recompile_limit", "cache_size_limit"):   # 28 layers x shape buckets
                if hasattr(dc, k):
                    setattr(dc, k, 1024)
            self.sl = torch.compile(stateq.state_layer, dynamic=False)
            self.ql = torch.compile(stateq.question_layer, dynamic=False)
        if flex:    # block-sparse attention in the state pass (flex compiles itself; do not wrap it): 168 -> ~30-38 ms
            from . import flexattn    # of attention at 2,000 tokens on a Jetson Orin, no gain below ~200 tokens
            self.sl = flexattn.state_layer_flex
        self.hits = self.misses = self.captures = 0

    # ---------------------------------------------------------------- graphs
    def _capture(self, key, fn, inputs):
        """fn(**inputs) -> output tensor(s); inputs are static buffers the caller fills before replay."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                fn(**inputs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.cuda.graph(g, pool=self.pool):
            out = fn(**inputs)
        self.captures += 1
        self.graphs[key] = (g, inputs, out)
        return self.graphs[key]

    def _state_graph(self, slot, Sp):
        key = ("s", slot, Sp)
        if key in self.graphs:
            return self.graphs[key]
        m, w = self.m, self.m.window
        kvb, hsb = self.kv[slot], self.hs[slot]
        # pos is a constant, but it must be held by self.graphs for the graph's lifetime (a fixed bug): it used to live only
        # in fn's closure, which is collected when this returns, so later replays read the freed buffer -> wrong K/V
        inp = {"ids": torch.full((1, Sp), self.st.tok.pad_token_id, device=self.dev, dtype=torch.long),
               "gm": torch.zeros(1, 1, Sp, Sp, dtype=torch.bool, device=self.dev),
               "lm": torch.zeros(1, 1, Sp, Sp, dtype=torch.bool, device=self.dev),
               "pos": torch.arange(Sp, device=self.dev)[None]}

        def fn(ids, gm, lm, pos):
            h = m.bert.embeddings(input_ids=ids)
            for i, lay in enumerate(m.bert.layers):
                h, k, v = self.sl(lay, h, pos, lm, gm)
                kvb[i, 0, :, :, :Sp].copy_(k)
                kvb[i, 1, :, :, :Sp].copy_(v)
            hsb[:, :Sp].copy_(m.bert.final_norm(h))
            return hsb
        return self._capture(key, fn, inp)

    def _question_graph(self, slot, Sp, Tp):
        key = ("q", slot, Sp, Tp)
        if key in self.graphs:
            return self.graphs[key]
        m, kvb = self.m, self.kv[slot]
        inp = {"ids": torch.full((1, Tp), self.st.tok.pad_token_id, device=self.dev, dtype=torch.long),
               "pos": torch.zeros(1, Tp, dtype=torch.long, device=self.dev),
               "gm": torch.zeros(1, 1, Tp, Sp + Tp, dtype=torch.bool, device=self.dev),
               "lm": torch.zeros(1, 1, Tp, Sp + Tp, dtype=torch.bool, device=self.dev)}

        def fn(ids, pos, gm, lm):
            x = m.bert.embeddings(input_ids=ids)
            for i, lay in enumerate(m.bert.layers):
                x = self.ql(lay, x, pos, kvb[i, 0, :, :, :Sp], kvb[i, 1, :, :, :Sp], gm, lm)
            return m.bert.final_norm(x)
        return self._capture(key, fn, inp)

    def reset(self):
        self.slot_of.clear()
        self.free = list(range(len(self.kv)))

    # ---------------------------------------------------------------- passes
    def encode_state(self, ids, key):
        S = ids.shape[1]
        if key in self.slot_of:
            self.slot_of.move_to_end(key)
            self.hits += 1
            return self.slot_of[key]
        self.misses += 1
        if self.free:
            slot = self.free.pop()
        else:
            _, (slot, _) = self.slot_of.popitem(last=False)
        Sp = _ceil(S, SB)
        g, inp, _ = self._state_graph(slot, Sp)
        inp["ids"].fill_(self.st.tok.pad_token_id)
        inp["ids"][:, :S].copy_(ids)
        p = torch.arange(Sp, device=self.dev)
        valid = p < S
        eye = torch.eye(Sp, dtype=torch.bool, device=self.dev)
        allowed = valid[:, None] & valid[None, :]
        inp["gm"].copy_((allowed | eye)[None, None])
        inp["lm"].copy_((allowed & ((p[:, None] - p[None, :]).abs() <= self.m.window // 2) | eye)[None, None])
        g.replay()
        self.slot_of[key] = (slot, S)
        return slot, S

    def forward(self, b):
        if b["ids"].shape[0] != 1:
            return self.full(b)
        blk, pos, ids = b["blk"][0], b["pos"][0], b["ids"]
        S = b["s_tok"].shape[1]
        key = tuple(ids[0, :S].tolist())
        slot, _ = self.encode_state(ids[:, :S], key)
        Sp = _ceil(S, SB)
        qsel = (blk[S:] > 0).nonzero()[:, 0] + S
        n = len(qsel)
        Tp = _ceil(n, QB)
        if self.m.lora_gate is not None:
            self.m.lora_gate.mask = None          # every token of the question pass is a question token
        g, inp, hq = self._question_graph(slot, Sp, Tp)
        qb, qp = blk[qsel], pos[qsel]
        w = self.m.window
        inp["ids"].fill_(self.st.tok.pad_token_id)
        inp["ids"][0, :n] = ids[0, qsel]
        inp["pos"].zero_()
        inp["pos"][0, :n] = qp
        sp_ = torch.arange(Sp, device=self.dev)
        s_ok = (sp_ < S)[None].expand(n, Sp)
        near_s = (qp[:, None] - sp_[None]).abs() <= w // 2
        same = qb[:, None] == qb[None, :]
        near_q = (qp[:, None] - qp[None, :]).abs() <= w // 2
        gm = torch.zeros(Tp, Sp + Tp, dtype=torch.bool, device=self.dev)
        lm = torch.zeros_like(gm)
        gm[:n, :Sp], gm[:n, Sp:Sp + n] = s_ok, same
        lm[:n, :Sp], lm[:n, Sp:Sp + n] = s_ok & near_s, same & near_q
        pad = torch.arange(n, Tp, device=self.dev)
        gm[pad, Sp + pad] = True
        lm[pad, Sp + pad] = True
        inp["gm"].copy_(gm[None, None])
        inp["lm"].copy_(lm[None, None])
        g.replay()
        T = ids.shape[1]
        row = torch.zeros(T, dtype=torch.long, device=self.dev)
        row[qsel] = torch.arange(n, device=self.dev)
        qt = hq[0][row[b["q_tok"] % T]]
        qm = b["q_tok_m"]
        qpool = (qt * qm[..., None]).sum(1) / qm.sum(1, keepdim=True)
        state = self.hs[slot][:, :S].expand(qt.shape[0], -1, -1)
        return self.m._heads(qpool, state, qt, qm)


def install(st, slots=3, compile=True, flex=False):
    sp = GSplit(st, slots, compile, flex)
    st.model.forward = sp.forward
    return sp
