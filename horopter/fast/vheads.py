# SPDX-License-Identifier: MIT
"""The heads of K adapters in ONE vectorised, CUDA-graph-capturable call, and the answer post-processing.

horopter.model.HoropterModel._heads + runtime.Student.answer's per-question maths (unrel, Answer.levels, softmaxes,
abstain-feature inputs), batched over K adapter slots with stacked weights, so a multi-adapter question pass costs
one launch sequence for all heads and one device-to-host copy for the whole request.

    hs  = HeadStack(st, K);  hs.fill(k, "typed")                  # or hs.fill(k, "typed", fb=flatbank)
    pl  = Plan.build(batches, groups, types, Tk, Nb, Mq, O)       # host side: static int/float buffers (CPU tensors)
    out = hs.heads_k(hq, state_hs, s_m, pl)                       # [K, Nq, W] fp32, call under autocast(bf16) on cuda
    answers = unpack(out.cpu().numpy(), pl.meta, s_m_np)          # [per k: {qid: {p, rel, unrel, f, ev_star}}]

Inputs of heads_k (all static-shape, so one graph per (K, Tk, Nb, Mq, Smax, Nq, O)):
    hq [K, Tk, d]      final-normed question-pass hidden states; segment k = adapter k's question/option blocks
                       concatenated in typed_pack order, then padding rows (any value, never read unmasked)
    state_hs [Smax, d] the cached final state hidden states (same for every adapter); Smax may be padded
    s_m [Smax]         1.0 on real state tokens (CLS state SEP), 0 on padding
    pl.tok [K,Nb,Mq]   row of hq (within its segment) of token m of block b      (int64)
    pl.qm  [K,Nb,Mq]   1.0 on the block's real tokens                            (fp32)
    pl.qb  [K,Nq]      block index of question n's question block               (int64)
    pl.opt [K,Nq,O]    block indices of its option blocks;   pl.om [K,Nq,O] bool validity of each option slot
    pl.typ [K,Nq]      0 noul / 1 choice / 2 score (index into tau)
Padded blocks/questions/options are ignored (pads have qm = 0, om = False; pad questions give finite garbage).

Packed output W = 4*O + 2 + 2*Smax, per (k, n):
    [0:O]            p (answer distribution over the options, 0 on padded slots)
    [O:4O]           rel [O, 3] softmax over support/refute/undetermined of each option block (row-major)
    [4O]             unrel = sigmoid(unrel_logit of the question block)
    [4O+1]           index of the argmax option (float)
    [4O+2 : 4O+2+2S] sigmoid(ev_s) of the argmax option, [Smax, 2] row-major (channel 0 SUPPORT, 1 CONFLICT)
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..model import TYPES, UNREL

D_MS = 128
D_LOC = 256


class HeadStack:
    """Stacked head parameters of K adapter slots. Fused buffers (views carry the per-parameter names):
        Ws  [K, 514, d]  state side:   ms_s | xs_s.0 | ev_u | xs_u.0 | loc_s        (rows 0:128, 128:256, 256, 257, 258:514)
        Wq  [K, 256, d]  question tokens: ms_q | xs_q.0
        Wp  [K, d+256, d] pooled block:   rel.0 | loc_q
    plus rel.2, qh.0, qh.2, ms_scale, xs_scale (fp32 or the live dtype) and tau [K, 3] fp32."""

    def __init__(self, st, K):
        m = st.model
        assert len(m.xs_s) == 1 and len(m.q_idx) == 3, "default CHANS only"
        p = m.rel[0].weight
        self.K, self.d, self.dt, self.dev = K, p.shape[1], p.dtype, p.device
        d, dt, dev = self.d, self.dt, self.dev
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
        self.Ws, self.bs = z(K, 2 * D_MS + 2 + D_LOC, d), z(K, 2 * D_MS + 2 + D_LOC)
        self.Wq, self.bq = z(K, 2 * D_MS, d), z(K, 2 * D_MS)
        self.Wp, self.bp = z(K, d + D_LOC, d), z(K, d + D_LOC)
        self.Wr2, self.br2 = z(K, 4, d), z(K, 4)
        self.Wh1, self.bh1 = z(K, 256, d), z(K, 256)
        self.Wh2, self.bh2 = z(K, 3, 256), z(K, 3)
        self.ms_scale, self.xs_scale = z(K), z(K)
        self.tau = torch.ones(K, len(TYPES), dtype=torch.float32, device=dev)
        S = slice
        # key -> (tensor, row slice or None): where each live parameter of one adapter lives in slot k
        self.slots = {
            "ms_s.weight": (self.Ws, S(0, 128)), "ms_s.bias": (self.bs, S(0, 128)),
            "xs_s.0.weight": (self.Ws, S(128, 256)), "xs_s.0.bias": (self.bs, S(128, 256)),
            "ev_u.weight": (self.Ws, S(256, 257)), "ev_u.bias": (self.bs, S(256, 257)),
            "xs_u.0.weight": (self.Ws, S(257, 258)), "xs_u.0.bias": (self.bs, S(257, 258)),
            "loc_s.weight": (self.Ws, S(258, 514)), "loc_s.bias": (self.bs, S(258, 514)),
            "ms_q.weight": (self.Wq, S(0, 128)), "ms_q.bias": (self.bq, S(0, 128)),
            "xs_q.0.weight": (self.Wq, S(128, 256)), "xs_q.0.bias": (self.bq, S(128, 256)),
            "rel.0.weight": (self.Wp, S(0, d)), "rel.0.bias": (self.bp, S(0, d)),
            "loc_q.weight": (self.Wp, S(d, d + 256)), "loc_q.bias": (self.bp, S(d, d + 256)),
            "rel.2.weight": (self.Wr2, None), "rel.2.bias": (self.br2, None),
            "qh.0.weight": (self.Wh1, None), "qh.0.bias": (self.bh1, None),
            "qh.2.weight": (self.Wh2, None), "qh.2.bias": (self.bh2, None),
            "ms_scale": (self.ms_scale, None), "xs_scale": (self.xs_scale, None), "answer.tau": (self.tau, None)}
        self.st, self.ar = st, torch.arange(K, device=dev)

    @torch.no_grad()
    def fill(self, k, name, fb=None):
        """Copy adapter `name`'s head tensors into slot k, from st.bank[name] or (fb = bank.FlatBank) its flat slices."""
        for key, (buf, sl) in self.slots.items():
            if fb is not None:
                dt, o, n, shape = fb.layout[key]
                t = fb.flat[name][dt][o:o + n].view(shape)
            else:
                t = self.st.bank[name][key]
            dst = buf[k] if sl is None else buf[k][sl]
            dst.copy_(t.reshape(dst.shape))

    # ------------------------------------------------------------------------------------------------ the heads
    @staticmethod
    def _lin(x, W, b):
        """Per-slot Linear: x [K|1, N, in] (a shared x is broadcast), W [K, out, in], b [K, out] -> [K, N, out]."""
        if x.shape[0] != W.shape[0]:
            x = x.expand(W.shape[0], -1, -1)
        return torch.baddbmm(b.unsqueeze(1), x, W.transpose(1, 2))

    def heads_k(self, hq, state_hs, s_m, pl, ev_q=False):
        """See the module docstring. Returns the packed [K, Nq, W] fp32 tensor (and ev_q [K, Nb, Mq, 3] if asked).
        Mirrors HoropterModel._heads op for op (same dtype promotions), vectorised over K."""
        K, Nb, Mq = pl.tok.shape
        L, S = self._lin, state_hs.shape[0]
        qt = hq[self.ar[:, None, None], pl.tok]                                  # [K, Nb, Mq, d]
        qm = pl.qm
        qpool = (qt * qm[..., None]).sum(2) / qm.sum(2, keepdim=True).clamp_min(1.0)   # [K, Nb, d]
        pp = L(qpool, self.Wp, self.bp)                                          # rel.0 | loc_q
        rel = L(F.gelu(pp[..., :self.d]), self.Wr2, self.br2).float()            # [K, Nb, 4]
        loc_q = pp[..., self.d:]                                                 # [K, Nb, 256]
        sp = L(state_hs[None], self.Ws, self.bs)                                 # [K, S, 514]
        qp = L(qt.reshape(K, Nb * Mq, -1), self.Wq, self.bq).view(K, Nb, Mq, 2 * D_MS)
        loc = (torch.einsum("ksc,kbc->kbs", sp[..., 258:], loc_q) / 16).float()  # [K, Nb, S]
        pad = (qm == 0)[:, :, None, :]                                           # [K, Nb, 1, Mq]
        chans = []
        for i, (u, scale) in enumerate(((256, self.ms_scale), (257, self.xs_scale))):
            s = F.normalize(sp[..., D_MS * i:D_MS * (i + 1)], dim=-1)
            q = F.normalize(qp[..., D_MS * i:D_MS * (i + 1)], dim=-1)
            sim = torch.einsum("ksc,kbmc->kbsm", s, q).masked_fill(pad, -2.0)
            ev = scale[:, None, None] * sim.max(-1).values + sp[..., u][:, None, :]
            chans.append((ev + loc).float())
        ev_s = torch.stack(chans, -1)                                            # [K, Nb, S, 2]
        # ---- answers
        ar = self.ar
        r_q = rel[ar[:, None], pl.qb]                                            # [K, Nq, 4]
        unrel = torch.sigmoid(r_q[..., UNREL] - torch.logsumexp(r_q[..., :UNREL], -1))
        r_o = rel[ar[:, None, None], pl.opt]                                     # [K, Nq, O, 4]
        lsm = F.log_softmax(r_o[..., :UNREL], -1)
        tau = self.tau[ar[:, None], pl.typ]                                      # [K, Nq]
        z = (tau[..., None] * lsm[..., 0]).masked_fill(~pl.om, float("-inf"))
        p = z.softmax(-1)
        rel3 = torch.where(pl.om[..., None], r_o[..., :UNREL].softmax(-1), 0.0)
        star = z.argmax(-1)                                                      # [K, Nq]
        blk = pl.opt.gather(2, star[..., None])[..., 0]
        ev_star = ev_s[ar[:, None], blk].sigmoid()                               # [K, Nq, S, 2]
        ev_star = ev_star * s_m[None, None, :, None]                             # padded state slots -> 0
        out = torch.cat([p, rel3.flatten(2), unrel[..., None], star[..., None].float(), ev_star.flatten(2)], -1)
        if ev_q:
            h = F.gelu(L(qt.reshape(K, Nb * Mq, -1), self.Wh1, self.bh1))
            return out, L(h, self.Wh2, self.bh2).float().view(K, Nb, Mq, 3)
        return out


# ------------------------------------------------------------------------------------------------------------- plan
@dataclass
class Plan:
    """Host-built, static-shape index tensors of one request (see module docstring). `meta[k]` = [(qid, type, n_opts)]."""
    tok: torch.Tensor
    qm: torch.Tensor
    qb: torch.Tensor
    opt: torch.Tensor
    om: torch.Tensor
    typ: torch.Tensor
    meta: list = field(default_factory=list)

    @staticmethod
    def build(block_lens, groups, items, Nb, Mq, O, Nq):
        """block_lens[k] = token count of each block of segment k, in order (b["q_tok_m"].sum(1)); groups[k] =
        typed_pack groups of segment k (block indices); items[k] = [(qid, question dict)] in the same order."""
        K = len(block_lens)
        tok, qm = torch.zeros(K, Nb, Mq, dtype=torch.long), torch.zeros(K, Nb, Mq)
        qb, opt = torch.zeros(K, Nq, dtype=torch.long), torch.zeros(K, Nq, O, dtype=torch.long)
        om, typ = torch.zeros(K, Nq, O, dtype=torch.bool), torch.zeros(K, Nq, dtype=torch.long)
        meta = []
        for k in range(K):
            o = 0
            for j, n in enumerate(block_lens[k]):
                tok[k, j, :n], qm[k, j, :n] = torch.arange(o, o + n), 1
                o += n
            meta.append([])
            for n, (g, (qid, q)) in enumerate(zip(groups[k], items[k])):
                qb[k, n], typ[k, n] = g["qb"], TYPES[q["type"]]
                opt[k, n, :len(g["opts"])], om[k, n, :len(g["opts"])] = torch.tensor(g["opts"]), True
                meta[k].append((qid, q["type"], len(g["opts"])))
        return Plan(tok, qm, qb, opt, om, typ, meta)

    def to(self, dev):
        return Plan(*(getattr(self, f).to(dev) for f in ("tok", "qm", "qb", "opt", "om", "typ")), self.meta)


# ----------------------------------------------------------------------------------------------------------- unpack
def _features(p, rel, u, qtype, ev):
    """abstain.features, from the packed pieces. ev = sigmoid evidence of the argmax option over the real state tokens
    [n, 2]; rel [k, 3] stance softmax of every option."""
    k = int(p.argmax())
    ps = np.sort(p)[::-1]
    p2 = float(ps[1]) if len(ps) > 1 else 0.0
    ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))
    es, er = ev[:, 0], ev[:, 1]
    nz = len(es) > 0
    return np.asarray([ps[0], p2, ps[0] - p2, ent, math.log(len(p)), *(float(qtype == t) for t in ("noul", "choice", "score")),
                       1 - u, *rel[k], rel[:, 0].max(), rel[:, 1].max(), rel[:, 2].mean(),
                       float(es.max()) if nz else 0.0, float(es.mean()) if nz else 0.0,
                       float(er.max()) if nz else 0.0, float(er.mean()) if nz else 0.0,
                       float((es > 0.5).mean()) if nz else 0.0, float((er > 0.5).mean()) if nz else 0.0], np.float32)


def unpack(packed, meta, s_m, O):
    """packed [K, Nq, W] numpy (the single .cpu() of the request), meta = Plan.meta, s_m numpy [Smax] ->
    [per k: {qid: {"p", "rel" [n,3], "unrel", "f" [21], "ev_star": (index, [Smax, 2] sigmoid evidence of that option)}}].
    Same keys and values as runtime.Student.answer, with "ev" replaced by the argmax option's evidence only."""
    S = len(s_m)
    real = s_m > 0
    res = []
    for k, qs in enumerate(meta):
        r = {}
        for n, (qid, typ, no) in enumerate(qs):
            v = packed[k, n]
            p = v[:no].astype(np.float32)
            rel = v[O:4 * O].reshape(O, 3)[:no]
            u = float(v[4 * O])
            star = int(v[4 * O + 1])
            ev = v[4 * O + 2:].reshape(S, 2)
            d = {"p": p, "rel": rel, "unrel": u, "ev_star": (star, ev)}
            d["f"] = _features(p, rel, u, typ, ev[real])
            r[qid] = d
        res.append(r)
    return res
