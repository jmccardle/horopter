# SPDX-License-Identifier: MIT
"""Multi-adapter question pass: the question side of a request as ONE CUDA graph replay + ONE device-to-host copy.

State pass and state cache are the single-adapter ones (graphq.GSplit, or stateq.Split on CPU): the state encoding is
adapter-blind. The question pass packs SEGMENTS of Tk tokens end to end into one sequence of N = K * Tk tokens:
    segment s    the question/option blocks of whole items of ONE adapter (an adapter may own several segments)
    LoRA         every trunk Linear computes base(x) + scale * bmm(bmm(x.view(K, Tk, in), A_s^T), B_s^T), A_s, B_s = the
                 adapter in stack slot s (no FLOPs on other adapters' deltas)
    masks        built INSIDE the graph from (bid, pos, s_m): a real token attends the cached state and its own block
                 (local layers: inside the window); pads attend only themselves; block ids differ across segments
    heads        the SAME graph continues with the K-adapter heads (vheads maths, rewritten for the graph): head slot k
                 holds the same adapter as LoRA slot k (one _fill writes both), MaxSim only on the argmax ("star")
                 option block, answer post-processing (softmax, unrel, argmax) and the evidence of the star block
    evidence     the evidence adapter's segment gathers at the FAMILY question's star (`ssrc` plan: slot*Nq+n of the
                 family row in the same pass); mode "all": the segments flagged `want_ev` sit in the LAST Ke slots and
                 return every option's evidence [Ke, Nq, O, Sp, 2]
    output       a packed fp32 result is copied into a persistent buffer allocated OUTSIDE the graph pool (never a pool
                 tensor), then ONE non_blocking copy to pinned host memory. Plan inputs: ONE int64 staging buffer
                 (ids, pos, bid, block starts/lens, qb, opt, om, typ, ssrc, s_m) and one H2D copy.
    graphs       one per (slot, Sp, K, Tk, O, Ke); Sp in SPB, K in KB, Tk in TKB (beyond 256 only for items too big for
                 256 tokens), O in OB, Ke 0 unless evidence mode "all". Nq = Tk // 8 items, Nb = Tk // 4 blocks and
                 Mq = min(Tk, max_q) tokens per block are DERIVED from Tk and enforced by the planner. Input/output
                 buffers are shared per shape across slots. One MultiSplit per model (asserted); graphs live in
                 self.graphs, not in gs.graphs, so they can never read a stale stack buffer.

    gs = bank.install(st, graphs=True)[0]; ms = multilora.MultiSplit(st, gs)        # CPU: bank.install(st); MultiSplit(st)
    ms.precapture(multilora.default_shapes())                                        # optional: capture at startup
    res = ms.answer_multi(state, {qid: (q, "typed")}, evidence=False)               # = runtime.Student.answer per qid
    res = multilora.serve_request(ms, state, qs, route, evidence)                    # the server's `with _LOCK:` body
    multilora.install_hook(st, ms)                                                   # same, with the unmodified server
evidence: False | True | "chosen" | "all". Answer dicts: {"p","rel","unrel","f"} (+ "ev" = list over options of
(offsets, [S, 2]) or None; in mode "chosen" only the argmax option of the family (evidence adapter: the family's
argmax) is filled, which is all the server reads).
"""
import collections

import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from . import graphq
from . import stateq
from . import vheads as V
from .. import abstain as A
from ..model import UNREL, LoRALinear
from ..packing import typed_pack

KB = (1, 2, 4, 8)                                   # segment-count buckets
TKB = (32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)   # segment length buckets (16 apart up to 128, then 32;
                                                    # beyond 256 steps of 64 up to 512, then 128)
NQ_DIV, NB_DIV = 16, 8                              # items per segment <= Tk // NQ_DIV, blocks <= Tk // NB_DIV
SPB = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)    # state-length buckets of the question graphs
OB = (4, 8, 16)                                     # option-count buckets


def _ceil(n, b):
    return -(-n // b) * b


def _bucket(n, bs=KB):
    return next((k for k in bs if k >= n), None)


def _tkb(n):
    return next((t for t in TKB if t >= n), _ceil(n, 64) if n <= 512 else _ceil(n, 128))


def _spb(n, smax):
    return next((s for s in SPB if s >= n and s <= smax), smax)


def _ob(n):
    return next((o for o in OB if o >= n), _ceil(n, 16))


def default_shapes(sps=(128, 192, 256, 384, 512, 768, 1024, 1536, 2048)):
    """The realistic capture set (Sp, K, Tk, O, Ke): K=1 and K=2 with up to 4 questions (Tk 64..128), K=4 as a
    stretch, K=8 (4 families + evidence) at Tk 64/128; O = 4 (noul/choice with up to 4 options)."""
    out = []
    for sp in sps:
        for K, tks in ((1, (64, 128, 256)), (2, (64, 128)), (4, (64, 128)), (8, (64, 128))):
            out += [(sp, K, tk, 4, 0) for tk in tks]
    return out


# ---------------------------------------------------------------------------------------------------- layer function
def _mlin(m, x, K):
    """Linear with the segmented multi-LoRA delta. x [1, K * Tk, in] -> [1, K * Tk, out]."""
    if not isinstance(m, LoRALinear):
        return m(x)
    a, b = m.mstack[K]
    d = torch.bmm(torch.bmm(x.reshape(K, -1, x.shape[-1]), a.transpose(1, 2)), b.transpose(1, 2)) * m.scale
    return m.base(x) + d.reshape(x.shape[0], -1, d.shape[-1])


def mquestion_layer(lay, x, pos, ks, vs, gm, lm, K):
    """stateq.question_layer with the K-segment LoRA (K static)."""
    at = lay.attn
    qkv = _mlin(at.Wqkv, lay.attn_norm(x), K).view(x.shape[0], -1, 3, at.num_heads, at.head_dim)
    cos, sin = at.rotary_emb(qkv, position_ids=pos)
    q, k, v = qkv.transpose(3, 1).unbind(dim=2)
    q, k = stateq.apply_rotary_pos_emb(q, k, cos, sin)
    m = lm if at.local_attention != (-1, -1) else gm
    a = stateq.F.scaled_dot_product_attention(q, torch.cat([ks, k], 2), torch.cat([vs, v], 2), attn_mask=m)
    x = x + _mlin(at.Wo, a.transpose(1, 2).reshape(x.shape[0], -1, at.all_head_size), K)
    mlp = lay.mlp
    i, g = _mlin(mlp.Wi, lay.mlp_norm(x), K).chunk(2, dim=-1)
    return x + _mlin(mlp.Wo, mlp.act(i) * g, K)


# ------------------------------------------------------------------------------------------------------------ planner
def _segments(by_job, cap):
    """by_job: [(job, [(item, ntok, nblk)])] -> [(job, [items])]: first-fit-decreasing under the caps of a Tk=cap segment
    (tokens, blocks <= cap // 4, items <= cap // 8). Items are never split."""
    segs = []
    for j, its in by_job:
        bins = []
        for i, n, nb in sorted(its, key=lambda x: -x[1]):
            b = next((b for b in bins if b[0] + n <= cap and b[1] + nb <= cap // NB_DIV and len(b[2]) < cap // NQ_DIV), None)
            if b is None:
                b = [0, 0, []]
                bins.append(b)
            b[0] += n
            b[1] += nb
            b[2].append(i)
        segs += [(j, sorted(b[2])) for b in bins]
    return segs


def plan(jobs_units, kmax=KB[-1]):
    """jobs_units[j] = [(ntok, nblk, key)] per item of job j (key = the question id: items with the same key, i.e. a
    question and its evidence copy, are always placed in the same pass). -> list of passes; a pass = [(job, [items])]
    with at most kmax segments, one Tk cap per pass chosen to minimise K_bucket * cap (then the segment count)."""
    allit = [(j, i, n, nb, k) for j, us in enumerate(jobs_units) for i, (n, nb, k) in enumerate(us)]

    def solve(sel):
        lo = max(max(n for _, _, n, _, _ in sel), NB_DIV * max(nb for _, _, _, nb, _ in sel), NQ_DIV)
        jobs = sorted({j for j, *_ in sel})
        by_job = [(j, [(i, n, nb) for jj, i, n, nb, _ in sel if jj == j]) for j in jobs]
        hi = max(sum(x[2] for x in sel if x[0] == j) for j in jobs)
        caps = [t for t in TKB if t >= lo]
        top = _ceil(max(hi, lo), 64)
        caps += [c for c in range(TKB[-1] + 64, min(top, 512) + 1, 64) if c >= lo]
        caps += [c for c in range(640, top + 1, 128) if c >= lo]
        if not caps:
            caps = [_tkb(lo)]
        best = None
        for cap in caps:
            segs = _segments(by_job, cap)
            if len(segs) > kmax:
                continue
            c = (_bucket(len(segs)) * cap, len(segs), cap)
            if best is None or c < best[0]:
                best = (c, segs)
        if best is not None:
            return [best[1]]
        keys = list(dict.fromkeys(k for *_, k in sel))
        if len(keys) > 1:
            h = len(keys) // 2
            a, bset = set(keys[:h]), set(keys[h:])
            return solve([x for x in sel if x[4] in a]) + solve([x for x in sel if x[4] in bset])
        h = len(sel) // 2
        return solve(sel[:h]) + solve(sel[h:])
    return solve(allit)


class _Shape:
    """Dims and buffers of one (Sp, K, Tk, O, Ke) shape, shared by every slot's graph of that shape."""

    def __init__(self, ms, key):
        self.key = key
        Sp, K, Tk, O, Ke = key
        self.Sp, self.K, self.Tk, self.O, self.Ke = key
        self.N, self.Nq, self.Nb = K * Tk, max(1, Tk // NQ_DIV), max(1, Tk // NB_DIV)
        self.Mq = min(Tk, ms.st.col.mq)
        self.W = 4 * O + 2 + 2 * Sp
        self.n_main, self.n_all = K * self.Nq * self.W, Ke * self.Nq * O * Sp * 2
        N, Nq, Nb = self.N, self.Nq, self.Nb
        fields = [("ids", N), ("pos", N), ("bid", N), ("bstart", K * Nb), ("blen", K * Nb), ("qb", K * Nq),
                  ("opt", K * Nq * O), ("om", K * Nq * O), ("typ", K * Nq), ("ssrc", K * Nq), ("s_m", Sp)]
        self.lay, o = {}, 0
        for nme, n in fields:
            self.lay[nme] = (o, n)
            o += n
        self.n_in = o
        cuda = ms.dev.type == "cuda"
        self.stage = torch.zeros(o, dtype=torch.long)
        if cuda:
            self.stage = self.stage.pin_memory()
        self.np = self.stage.numpy()
        self.dev = torch.zeros(o, dtype=torch.long, device=ms.dev)
        self.out = torch.zeros(self.n_main + self.n_all, dtype=torch.float32, device=ms.dev)   # outside the graph pool
        self.host = torch.zeros(self.n_main + self.n_all, dtype=torch.float32)
        if cuda:
            self.host = self.host.pin_memory()
        self.hnp = self.host.numpy()
        self.blank = np.zeros(o, np.int64)
        self.view_np(self.blank, "bid")[:] = -1
        self.view_np(self.blank, "om").reshape(K, Nq, O)[..., 0] = 1     # pad questions: one valid option (no NaN)
        self.view_np(self.blank, "ids")[:] = ms.st.tok.pad_token_id
        self.view_np(self.blank, "ssrc")[:] = np.arange(K * Nq)             # pad rows read only themselves
        self.I = {n: self.dev[a:a + c] for n, (a, c) in self.lay.items()}
        self.dev.copy_(torch.from_numpy(self.blank))

    def view_np(self, arr, name):
        a, c = self.lay[name]
        return arr[a:a + c]

    def cview(self, name):
        return self.view_np(self.np, name)


class MultiSplit:
    def __init__(self, st, gs=None, compile=None, kmax=KB[-1]):
        if compile is None:      # follow the graph state's setting (uncompiled, this pass costs ~40% more than graphq's)
            compile = gs is not None and gs.sl is not stateq.state_layer
        assert kmax in KB, f"kmax must be one of {KB}"
        self.st, self.m, self.gs, self.kmax = st, st.model, gs, kmax
        if getattr(st.model, "_multisplit", None) is not None:
            raise RuntimeError("one MultiSplit per model: the LoRALinears hold views of its stack buffers)")
        fb = getattr(st.load_adapter, "__self__", None)
        if fb is None or not hasattr(fb, "layout"):
            raise ValueError("install bank.FlatBank(st) (bank.install) first")
        self.fb = fb
        self.sp = None if gs else stateq.Split(self.m)            # CPU / eager: the plain state cache
        self.dev = next(self.m.parameters()).device
        # LoRA region of each flat dtype buffer (the bert.* keys are contiguous after sorting)
        self.region = {}
        for k, (dt, o, n, _) in fb.layout.items():
            if k.startswith("bert."):
                lo, hi = self.region.get(dt, (o, o + n))
                self.region[dt] = (min(lo, o), max(hi, o + n))
        self.region = {dt: (lo, hi - lo) for dt, (lo, hi) in self.region.items()}
        self.big = {dt: torch.zeros(kmax, n, dtype=dt, device=self.dev) for dt, (_, n) in self.region.items()}
        self.hs = V.HeadStack(st, kmax)
        self.ar = torch.arange(kmax, device=self.dev)
        self.slot_ad = [None] * kmax                              # adapter currently in each stack slot (LoRA + heads)
        for n, mod in self.m.bert.named_modules():
            if isinstance(mod, LoRALinear):
                assert not hasattr(mod, "mstack")
                views = {}
                for K in KB:
                    if K > kmax:
                        continue
                    ab = []
                    for t in ("A", "B"):
                        dt, o, num, shape = fb.layout[f"bert.{n}.{t}"]
                        o0 = self.region[dt][0]
                        ab.append(self.big[dt][:K, o - o0:o - o0 + num].view(K, *shape))
                    views[K] = tuple(ab)
                mod.mstack = views
        self.m._multisplit = self
        self.ql = torch.compile(mquestion_layer, dynamic=False) if compile else mquestion_layer
        self.shapes, self.graphs, self._hk = {}, {}, {}
        self.passes = self.graph_hits = self.captures = 0
        self.stats = collections.Counter()
        self.dbg = None                       # tests: a dict that eager passes fill with intermediates
        self.profile, self.replay_ms = False, []
        self.last_shape = None
        self.heads_off = False
        self.dry = None                         # set() -> run() only records the shape keys it would use
        self.force_tk = 0                       # profiling: lower bound of the Tk bucket

    def close(self):
        for mod in self.m.modules():
            if isinstance(mod, LoRALinear) and hasattr(mod, "mstack"):
                del mod.mstack
        self.m._multisplit = None
        self.graphs.clear()

    # --------------------------------------------------------------------------------------------- adapters / stacks
    def _flat(self, name):
        f = self.fb.flat
        if name not in f:
            f[name] = self.fb.pack(self.st.bank[name])         # an adapter added after FlatBank
        return f[name]

    @torch.no_grad()
    def _fill(self, slot, name):
        """Slot `slot` of the LoRA stacks AND of the head stacks := adapter `name` (the heads cannot see another adapter
        than the trunk)."""
        flat = self._flat(name)
        for dt, buf in self.big.items():
            o0, n = self.region[dt]
            buf[slot].copy_(flat[dt][o0:o0 + n])
        self.hs.fill(slot, name, fb=self.fb)
        self.slot_ad[slot] = name

    def _assign(self, ads, evf, K, Ke):
        """Stack slot of each segment: reuse a slot that already holds the adapter; `evf` segments go to the last
        Ke slots. Returns (slot per segment, [(slot, adapter) to fill])."""
        taken, out = set(), {}
        groups = (([s for s in range(len(ads)) if not (evf[s] and Ke)], range(0, K - Ke)),
                  ([s for s in range(len(ads)) if evf[s] and Ke], range(K - Ke, K)))
        for grp, rng in groups:
            for s in grp:
                t = next((t for t in rng if t not in taken and self.slot_ad[t] == ads[s]), None)
                if t is not None:
                    out[s] = t
                    taken.add(t)
            for s in grp:
                if s not in out:
                    t = next(t for t in rng if t not in taken)
                    out[s] = t
                    taken.add(t)
        return [out[s] for s in range(len(ads))]

    def hk(self, k0, k1):
        """Head stacks of slots [k0, k1) as views (static addresses)."""
        if (k0, k1) not in self._hk:
            h, S = self.hs, slice(k0, k1)
            self._hk[(k0, k1)] = SimpleNamespace(**{n: getattr(h, n)[S] for n in (
                "Ws", "bs", "Wq", "bq", "Wp", "bp", "Wr2", "br2", "ms_scale", "xs_scale", "tau")})
        return self._hk[(k0, k1)]

    # ---------------------------------------------------------------------------------------------------- compute
    def _ev(self, hk, sp, qt, qm, locq):
        """Evidence of N blocks per slot. qt [K,N,Mq,d], qm [K,N,Mq] fp32, locq [K,N,256] -> sigmoid-free logits [K,N,Sp,2]."""
        K, N, Mq, _ = qt.shape
        qp = V.HeadStack._lin(qt.reshape(K, N * Mq, -1), hk.Wq, hk.bq).view(K, N, Mq, 2 * V.D_MS)
        loc = (torch.einsum("ksc,knc->kns", sp[..., 258:], locq) / 16).float()
        if self.dbg is not None:
            self.dbg.setdefault("loc", []).append(loc)
        pad = (qm == 0)[:, :, None, :]
        chans = []
        for i, (u, scale) in enumerate(((256, hk.ms_scale), (257, hk.xs_scale))):
            s = F.normalize(sp[..., V.D_MS * i:V.D_MS * (i + 1)], dim=-1)
            q = F.normalize(qp[..., V.D_MS * i:V.D_MS * (i + 1)], dim=-1)
            sim = torch.einsum("ksc,knmc->knsm", s, q).masked_fill(pad, -2.0)
            ev = scale[:, None, None] * sim.max(-1).values + sp[..., u][:, None, :]
            chans.append((ev + loc).float())
        return torch.stack(chans, -1)

    def _compute(self, sh, I, kvf, shs, dbg=None):
        """Trunk + heads for one pass. I: views of the plan buffer; kvf(i) -> (ks, vs) of the cached state; shs [Sp, d]
        the cached final state hidden states. -> flat fp32 [n_main + n_all]."""
        m, dev = self.m, self.dev
        K, Tk, Nq, Nb, Mq, O, Ke, Sp, N = sh.K, sh.Tk, sh.Nq, sh.Nb, sh.Mq, sh.O, sh.Ke, sh.Sp, sh.N
        w = m.window
        ids, pos, bid = I["ids"].view(1, N), I["pos"].view(1, N), I["bid"]
        valid = bid >= 0
        sp_ = torch.arange(Sp, device=dev)
        s_ok = valid[:, None] & (I["s_m"] > 0)[None]
        p0 = pos[0]
        near_s = (p0[:, None] - sp_[None]).abs() <= w // 2
        same = (bid[:, None] == bid[None, :]) & valid[:, None] & valid[None, :]
        near_q = (p0[:, None] - p0[None, :]).abs() <= w // 2
        eye = torch.eye(N, dtype=torch.bool, device=dev)
        gm = torch.cat([s_ok, same | eye], 1)[None, None]
        lm = torch.cat([s_ok & near_s, (same & near_q) | eye], 1)[None, None]
        x = m.bert.embeddings(input_ids=ids)
        for i, lay in enumerate(m.bert.layers):
            ks, vs = kvf(i)
            x = self.ql(lay, x, pos, ks, vs, gm, lm, K)
        hq = m.bert.final_norm(x)[0].view(K, Tk, -1)
        if self.heads_off:                                   # profiling only: trunk without heads
            o = torch.zeros(sh.n_main + sh.n_all, device=dev) + hq.float().mean() * 0
            o[:sh.n_main].view(sh.K, sh.Nq, sh.W)[..., 0] = 1.0      # a valid answer (p = [1, 0, ...])
            return o
        return self._heads(sh, I, hq, shs, dbg)

    def _heads(self, sh, I, hq, shs, dbg=None):
        K, Tk, Nq, Nb, Mq, O, Ke, Sp = sh.K, sh.Tk, sh.Nq, sh.Nb, sh.Mq, sh.O, sh.Ke, sh.Sp
        dev, L = self.dev, V.HeadStack._lin
        hk, ar = self.hk(0, K), self.ar[:K]
        d = hq.shape[-1]
        bs, bl = I["bstart"].view(K, Nb), I["blen"].view(K, Nb)
        qb, typ, ssrc = I["qb"].view(K, Nq), I["typ"].view(K, Nq), I["ssrc"].view(K, Nq)
        opt, om = I["opt"].view(K, Nq, O), I["om"].view(K, Nq, O).bool()
        t = torch.arange(Tk, device=dev)
        pm = ((t >= bs[..., None]) & (t < (bs + bl)[..., None])).float()           # [K, Nb, Tk] block membership
        with torch.autocast(device_type=dev.type, enabled=False):
            qsum = torch.bmm(pm, hq.float())                                       # exact fp32 sums of the block rows
        qpool = qsum / bl.clamp(min=1)[..., None]
        pp = L(qpool, hk.Wp, hk.bp)                                                # rel.0 | loc_q
        rel = L(F.gelu(pp[..., :d]), hk.Wr2, hk.br2).float()                       # [K, Nb, 4]
        sp = L(shs[None], hk.Ws, hk.bs)                                            # [K, Sp, 514]
        r_q = rel[ar[:, None], qb]
        unrel = torch.sigmoid(r_q[..., UNREL] - torch.logsumexp(r_q[..., :UNREL], -1))
        r_o = rel[ar[:, None, None], opt]                                          # [K, Nq, O, 4]
        lsm = F.log_softmax(r_o[..., :UNREL], -1)
        tau = hk.tau[ar[:, None], typ]
        z = (tau[..., None] * lsm[..., 0]).masked_fill(~om, float("-inf"))
        p = z.softmax(-1)
        rel3 = torch.where(om[..., None], r_o[..., :UNREL].softmax(-1), 0.0)
        star0 = p.argmax(-1)                                                       # [K, Nq]
        star = star0.reshape(-1)[ssrc.reshape(-1)].view(K, Nq)                     # evidence rows: the family's star
        blk = opt.gather(2, star[..., None])[..., 0]                               # block of the star option
        m_ = torch.arange(Mq, device=dev)
        sb, sl = bs[ar[:, None], blk], bl[ar[:, None], blk]
        rows = (sb[..., None] + m_).clamp(max=Tk - 1)
        qt = hq[ar[:, None, None], rows]                                           # [K, Nq, Mq, d]
        qm = (m_ < sl[..., None]).float()
        locq = pp[ar[:, None], blk][..., d:]
        evl = self._ev(hk, sp, qt, qm, locq)                                       # [K, Nq, Sp, 2]
        s_m = (I["s_m"] > 0).float()
        ev_star = evl.sigmoid() * s_m[None, None, :, None]
        if dbg is not None:
            dbg.update(evl=evl, sp=sp, rel=rel, star=star, hq=hq)
        main = torch.cat([p, rel3.flatten(2), unrel[..., None], star[..., None].float(), ev_star.flatten(2)], -1)
        parts = [main.flatten()]
        if Ke:                                                                     # every option's evidence, last Ke slots
            k0 = K - Ke
            hk2, ar2 = self.hk(k0, K), self.ar[:Ke]
            opt2 = opt[k0:].reshape(Ke, Nq * O)
            sb2, sl2 = bs[k0:][ar2[:, None], opt2], bl[k0:][ar2[:, None], opt2]
            rows2 = (sb2[..., None] + m_).clamp(max=Tk - 1)
            qt2 = hq[k0:][ar2[:, None, None], rows2]
            qm2 = (m_ < sl2[..., None]).float()
            locq2 = pp[k0:][ar2[:, None], opt2][..., d:]
            ev2 = self._ev(hk2, sp[k0:], qt2, qm2, locq2).sigmoid() * s_m[None, None, :, None]
            parts.append(ev2.flatten())
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    # ---------------------------------------------------------------------------------------------------- graphs
    def _shape(self, key):
        if key not in self.shapes:
            self.shapes[key] = _Shape(self, key)
        return self.shapes[key]

    def _graph(self, slot, key):
        k = (slot, *key)
        if k in self.graphs:
            self.graph_hits += 1
            return self.graphs[k]
        gs, sh = self.gs, self._shape(key)
        kvb, shs = gs.kv[slot], gs.hs[slot][0, :sh.Sp]
        kvf = lambda i: (kvb[i, 0, :, :, :sh.Sp], kvb[i, 1, :, :, :sh.Sp])

        def fn():
            sh.out.copy_(self._compute(sh, sh.I, kvf, shs))     # the result leaves the pool through a persistent buffer
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.cuda.graph(g, pool=gs.pool):
            fn()
        self.captures += 1
        self.graphs[k] = g
        return g

    def precapture(self, shapes, slots=None):
        """Capture the graphs of `shapes` ((Sp, K, Tk, O, Ke) tuples, see default_shapes) for every state slot, so the
        first request of a shape does not pay the capture. Returns the number of graphs captured."""
        if self.gs is None:
            return 0
        n0 = self.captures
        for key in shapes:
            for slot in (range(len(self.gs.kv)) if slots is None else slots):
                self._graph(slot, key)
        return self.captures - n0

    def discover(self, requests):
        """Shape keys (Sp, K, Tk, O, Ke) a list of requests [(state, qs, route, evidence)] needs (no graph is captured; the
        state pass of each state is run). Use it to build the precapture list of a deployment."""
        self.dry = set()
        try:
            for state, qs, route, ev in requests:
                serve_request(self, state, qs, route, ev)
            return sorted(self.dry)
        finally:
            self.dry = None

    def graph_count(self):
        return len(self.graphs)

    def _execute(self, sh, slot, kv, hs_rows):
        """Replay (or eagerly run) one pass whose plan is in sh.np. -> numpy flat result."""
        if self.dry is not None:                       # shape discovery: record the key, run nothing
            self.dry.add(sh.key)
            return np.zeros(sh.n_main + sh.n_all, np.float32)
        self.passes += 1
        if self.gs is not None:
            sh.dev.copy_(sh.stage, non_blocking=True)
            g = self._graph(slot, sh.key)
            if self.profile:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                self.replay_ms.append((e0, e1))
            else:
                g.replay()
            sh.host.copy_(sh.out, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return sh.hnp
        sh.dev.copy_(sh.stage)
        kvf = lambda i: kv[i]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.st.autocast):
            sh.out.copy_(self._compute(sh, sh.I, kvf, hs_rows, self.dbg))
        sh.host.copy_(sh.out)
        return sh.hnp

    # ---------------------------------------------------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, state, jobs, evidence=False):
        """jobs: [(adapter, [(qid, q)])] or [(adapter, items, want_ev)] (an adapter may appear in several jobs). Jobs
        named "evidence" take the family question's argmax option for their evidence. -> [{qid: answer dict}] per job."""
        st, m, tok = self.st, self.m, self.st.tok
        mode = {False: None, None: None, True: "chosen"}.get(evidence, evidence)
        assert mode in (None, "chosen", "all"), evidence
        jobs = [(j[0], list(j[1]), bool(j[2]) if len(j) > 2 else mode is not None) for j in jobs]
        out = [{} for _ in jobs]
        if not any(it for _, it, _ in jobs):
            return out
        m.eval()
        if m.lora_gate is not None:
            m.lora_gate.mask = None                  # every token of the question pass is a question token
        uq = {}
        for _, items, _ in jobs:
            for qid, q in items:
                if uq.setdefault(qid, q) != q:
                    raise ValueError(f"question id {qid!r} used with two different questions in one request")
        qids = list(uq)
        pack, groups = typed_pack(state, [{"row": "req", "qid": i, "q": uq[i], "type": uq[i]["type"], "target": None,
                                           "gold": None} for i in qids])
        b = st.col([pack])
        ids, pos, blk = b["ids"], b["pos"][0].numpy(), b["blk"][0].numpy()
        S = b["s_tok"].shape[1]
        key = tuple(ids[0, :S].tolist())
        if self.gs is not None:
            slot, _ = self.gs.encode_state(ids[:, :S].to(self.dev), key)
            Sp = _spb(_ceil(S, graphq.SB), self.gs.Smax)
            kv, shs = None, None
        else:
            slot = None
            kv, hs0 = self.sp.encode_state(ids[:, :S].to(self.dev), key)
            Sp, shs = S, hs0[0]
        ids = ids[0].numpy()
        span = {}                                    # qid -> (lo, hi, [(block start rel. to lo, block length)], n_opts)
        for qid, g in zip(qids, groups):
            bl = range(g["qb"], (g["opts"][-1] if g["opts"] else g["qb"]) + 1)
            st_ = np.searchsorted(blk, [x + 1 for x in bl], "left")
            en_ = np.searchsorted(blk, [x + 1 for x in bl], "right")
            span[qid] = (int(st_[0]), int(en_[-1]), [(int(a - st_[0]), int(e - a)) for a, e in zip(st_, en_)], len(g["opts"]))
        units = [[(span[q][1] - span[q][0], len(span[q][2]), q) for q, _ in items] for _, items, _ in jobs]
        offs = None
        for pas in plan(units, self.kmax):
            segs = [s for s in pas if s[1]]
            ev_f = [jobs[j][2] for j, _ in segs]
            nev = sum(ev_f) if mode == "all" else 0
            K = _bucket(len(segs))
            Tk = _tkb(max(max(sum(units[j][i][0] for i in its) for j, its in segs),
                          NB_DIV * max(sum(units[j][i][1] for i in its) for j, its in segs),
                          NQ_DIV * max(len(its) for _, its in segs), self.force_tk))
            O = _ob(max(span[jobs[j][1][i][0]][3] for j, its in segs for i in its))
            sh = self._shape((Sp, K, Tk, O, nev))
            self.last_shape = sh
            slots = self._assign([jobs[j][0] for j, _ in segs], ev_f, K, nev)
            for s, (j, _) in zip(slots, segs):
                if self.slot_ad[s] != jobs[j][0]:
                    self._fill(s, jobs[j][0])
            np.copyto(sh.np, sh.blank)
            Nq, Nb = sh.Nq, sh.Nb
            f = {n: sh.cview(n) for n in sh.lay}
            f_ids, f_pos, f_bid = f["ids"], f["pos"], f["bid"]
            f_bs, f_bl = f["bstart"].reshape(K, Nb), f["blen"].reshape(K, Nb)
            f_qb, f_typ, f_ss = f["qb"].reshape(K, Nq), f["typ"].reshape(K, Nq), f["ssrc"].reshape(K, Nq)
            f_opt, f_om = f["opt"].reshape(K, Nq, O), f["om"].reshape(K, Nq, O)
            f["s_m"][:S] = 1
            nblk = int(blk.max()) + 1
            rows, fam, link = [], {}, []
            for s, (j, its) in zip(slots, segs):
                o, nb_, n = s * Tk, 0, 0
                for i in its:
                    qid, q = jobs[j][1][i]
                    lo, hi, bls, no = span[qid]
                    ln = hi - lo
                    f_ids[o:o + ln], f_pos[o:o + ln] = ids[lo:hi], pos[lo:hi]
                    f_bid[o:o + ln] = blk[lo:hi] + s * nblk
                    for bst, bln in bls:
                        f_bs[s, nb_], f_bl[s, nb_] = o - s * Tk + bst, bln
                        nb_ += 1
                    f_qb[s, n] = nb_ - len(bls)
                    f_opt[s, n, :no] = np.arange(nb_ - no, nb_)
                    f_om[s, n, :no] = 1
                    f_typ[s, n] = {"noul": 0, "choice": 1, "score": 2}[q["type"]]
                    f_ss[s, n] = s * Nq + n
                    if jobs[j][0] == "evidence":
                        link.append((s, n, qid))
                    else:
                        fam.setdefault(qid, s * Nq + n)
                    rows.append((s, n, j, qid, q["type"], no, jobs[j][2]))
                    o += ln
                    n += 1
            for s, n, qid in link:
                if qid in fam:
                    f_ss[s, n] = fam[qid]
            self.last_rows = [(s_, n_, jobs[j_][0], q_) for s_, n_, j_, q_, *_ in rows]
            flat = self._execute(sh, slot, kv, shs)
            mainv = flat[:sh.n_main].reshape(K, Nq, sh.W)
            allv = flat[sh.n_main:].reshape(nev, Nq, O, Sp, 2) if nev else None
            for s, n, j, qid, typ, no, wev in rows:
                v = mainv[s, n]
                p = v[:no].astype(np.float32)
                rel = np.array(v[O:4 * O].reshape(O, 3)[:no])
                u = float(v[4 * O])
                star = int(v[4 * O + 1])
                evs = v[4 * O + 2:].reshape(Sp, 2)[:S]
                d = {"p": p, "rel": rel, "unrel": u, "f": V._features(p, rel, u, typ, evs)}
                if wev:
                    if offs is None:
                        offs = st.state_offsets(state)
                    if allv is not None and s >= K - nev:
                        d["ev"] = [(offs, np.array(allv[s - (K - nev), n, o_, :S])) for o_ in range(no)]
                    else:
                        d["ev"] = [None] * no
                        d["ev"][star] = (offs, np.array(evs))
                out[j][qid] = d
        return out

    def answer_multi(self, state, qs, evidence=False):
        """qs: {qid: (question, adapter)} -> {qid: answer dict} as runtime.Student.answer."""
        by = collections.defaultdict(list)
        for qid, (q, ad) in qs.items():
            by[ad].append((qid, q))
        ads = sorted(by)
        res = {}
        for r in self.run(state, [(a, by[a]) for a in ads], evidence):
            res.update(r)
        return res


# ------------------------------------------------------------------------------------------------------- server hooks
def serve_request(ms, state, qs, route, evidence, bank=None):
    """The body of the server's `with _LOCK:` block after routing. route: {qid: {"adapter": name, ...}}; evidence:
    False | True | "chosen" | "all" (the server's mbnli.evidence.mode). The evidence adapter (if loaded and evidence is on)
    is more segments of the same pass: its spans replace the family's. One graph replay + one D2H copy per pass."""
    bank = bank or ms.st.bank
    by = collections.defaultdict(list)
    for qid in qs:
        by[route[qid]["adapter"]].append((qid, qs[qid]))
    ev_ad = "evidence" in bank and bool(evidence)
    jobs = [(a, by[a], bool(evidence) and not ev_ad) for a in sorted(by)]
    if ev_ad:
        jobs.append(("evidence", list(qs.items()), True))
    outs = ms.run(state, jobs, evidence=evidence)
    res = {}
    for o in outs[: len(by)]:
        res.update(o)
    if ev_ad:
        for qid, d in outs[-1].items():
            res[qid]["ev"] = d["ev"]
    return res


class _Lazy(dict):
    """An answer that is filled when first touched, so the unmodified server can queue every st.answer call of a request
    and the whole request runs as one multi-adapter pass. (json.dumps / dict(d) read the C storage directly: call
    `_fill()` first, or index a key.)"""
    def __init__(self, hook):
        super().__init__()
        self.hook = hook

    def _fill(self):
        if not self.hook.done:
            self.hook.flush()


def _wrap(name):
    f = getattr(dict, name)

    def g(self, *a, **k):
        self._fill()
        return f(self, *a, **k)
    g.__name__ = name
    return g


for _n in ("__getitem__", "get", "__contains__", "keys", "items", "values", "__iter__", "__len__", "copy", "pop",
           "setdefault", "__repr__", "__eq__"):
    setattr(_Lazy, _n, _wrap(_n))


class _Hook:
    ev_mode = "all"          # the hook cannot see the server's mode: all options' evidence is always correct

    def __init__(self, st, ms):
        self.st, self.ms = st, ms
        self.use0, self.answer0 = st.use, st.answer
        self.cur, self.queue, self.done = None, [], True
        st.use, st.answer = self.use, self.answer

    def use(self, name):
        if name == "router":
            return self.use0(name)
        self.cur = name                      # an adapter pass is deferred: no weight copy now

    def answer(self, state, questions, evidence=False, max_blocks=96):
        if self.cur is None:
            return self.answer0(state, questions, evidence, max_blocks)
        if self.done:
            self.queue, self.done = [], False
        res = {qid: _Lazy(self) for qid in questions}
        self.queue.append((state, self.cur, dict(questions), evidence, res))
        return res

    def flush(self):
        q, self.queue, self.done = self.queue, [], True
        try:
            for state in dict.fromkeys(x[0] for x in q):
                part = [x for x in q if x[0] == state]
                outs = self.ms.run(state, [(ad, list(qs.items()), bool(ev)) for _, ad, qs, ev, _ in part],
                                   evidence=self.ev_mode if any(x[3] for x in part) else False)
                for (_, ad, qs, ev, res), o in zip(part, outs):
                    for qid, d in o.items():
                        dict.update(res[qid], d)
        except BaseException:
            self.queue, self.done = q, False
            raise
        self.cur = None


def install_hook(st, ms):
    """Make the unmodified horopter.server use the multi-adapter pass: st.use/st.answer calls of one request are queued and
    run together on the first read of an answer. The router (st.use('router') + answer_one_each) is untouched."""
    return _Hook(st, ms)
