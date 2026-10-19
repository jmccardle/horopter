# SPDX-License-Identifier: MIT
"""The Horopter trunk ("design B"), its heads, the typed answer head and question-path LoRA.

Design B: a batch is B packs; pack b = one state + n_b question blocks. In every layer the question tokens read the
state (packed block mask) while the state is question-blind, so the state encoding is the same whatever is asked and
can be cached. Every head reads only the final state tokens and this question's own tokens.

Heads (on the pooled question block / the state tokens):
  rel     4-way relation: support / refute / undetermined / unrelated
  ev_s    per-state-token evidence channels [Nq, Smax, Cs]: SUPPORT and CONFLICT (served as "refute"). Channel 0 is a
          ColBERT-style MaxSim head (+ a unary term and the shared region logit), the others are MaxSim heads of their
          own with the same region logit added
  ev_q    per-question-token channels [Nq, Mq, Cq]: SUPPORT, CONFLICT, UNSUPPORTED (an MLP over the question tokens)
  loc     region (localisation) logit over the state tokens

Answer (the "hard" arm): option logit = tau_type * log p(support) under the 3-way stance softmax of the option block
(unrelated excluded), so the answer is the normalised support and relevance stays out of the answer.

LoRA: apply_lora wraps every nn.Linear in the trunk layers with W x + (alpha / r) B A x (B = 0 at init). With a
LoRAGate the update is multiplied by a per-token mask (1 on question tokens), so state tokens see the frozen trunk
exactly and one state encoding is valid for every adapter.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel

N_REL = 4
TYPES = {"noul": 0, "choice": 1, "score": 2}
SUP, REF, UND, UNREL = 0, 1, 2, 3   # relation logits; also the state channel indices SUPPORT, CONFLICT
# evidence channels: (name, side, kind, labels); side s = state tokens, q = question tokens
CHANS = [("SUPPORT", "s", "role", (0,)), ("CONFLICT", "s", "role", (1,)),
         ("SUPPORT", "q", "role", (0,)), ("CONFLICT", "q", "role", (1,)), ("UNSUPPORTED", "q", "role", (2,))]
S_CH = [c[0] for c in CHANS if c[1] == "s"]


# ----------------------------------------------------------------------------------------------------------- LoRA
class LoRAGate:
    """Shared, mutable per-token mask for question-path LoRA; None = apply everywhere."""
    def __init__(self):
        self.mask = None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, gate: LoRAGate | None = None):
        super().__init__()
        self.base, self.gate = base, gate
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        d = (x @ self.A.t() @ self.B.t()) * self.scale
        if self.gate is not None and self.gate.mask is not None:
            d = d * self.gate.mask
        return self.base(x) + d


@torch.no_grad()
def merge_lora(bert: nn.Module) -> int:
    """Fold every (ungated) LoRALinear under bert.layers into its base Linear and unwrap it. Returns the count."""
    n = 0
    for mod in list(bert.layers.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                if child.gate is not None:
                    raise ValueError("a question-path (gated) LoRA can't be merged: the state path must see the base")
                child.base.weight += (child.B @ child.A).to(child.base.weight.dtype) * child.scale
                setattr(mod, name, child.base)
                n += 1
    return n


def apply_lora(bert: nn.Module, r: int, alpha: float | None = None, gate: LoRAGate | None = None) -> list[nn.Parameter]:
    """Freeze `bert`, add LoRA to the Linear layers under bert.layers; returns the trainable LoRA parameters."""
    alpha = 2 * r if alpha is None else alpha
    for p in bert.parameters():
        p.requires_grad_(False)
    targets = [(mod, name, child) for mod in bert.layers.modules() for name, child in mod.named_children()
               if isinstance(child, nn.Linear)]
    params = []
    for mod, name, child in targets:
        w = LoRALinear(child, r, alpha, gate)
        setattr(mod, name, w)
        params += [w.A, w.B]
    return params


# ---------------------------------------------------------------------------------------------------------- trunk
def masks(blk, pos, window):
    """Design-B masks. blk/pos [X,T] (blk -1 = pad, 0 = state, >0 = question). A token may attend within its own block,
    and a question token may also attend to the state. Returns bool (global, local) masks [X,1,T,T]; True = may
    attend; local adds the sliding window (|q - k| <= window // 2)."""
    r, c = blk[:, :, None], blk[:, None, :]
    valid = (r >= 0) & (c >= 0)
    same = valid & (r == c)
    allowed = same | ((r > 0) & (c == 0))
    dpos = pos[:, :, None] - pos[:, None, :]
    near = dpos.abs() <= window // 2
    local = allowed & near
    eye = torch.eye(blk.shape[1], dtype=torch.bool, device=blk.device)[None]
    return (allowed | eye)[:, None], (local | eye)[:, None]


class HoropterModel(nn.Module):
    """Ettin (ModernBERT) encoder in design B, with the relation, evidence and region heads.

    base:   a Hugging Face model id (pretrained weights are downloaded: training from scratch), or
    config: a transformers config (weights come from a state dict afterwards: inference / export).
    Parameter names (bert.*, rel.*, ms_*, xs_*, qh.*, ...) match the research checkpoints."""

    def __init__(self, base=None, config=None, chans=CHANS, grad_ckpt=True):
        super().__init__()
        if config is not None:
            config.reference_compile = False
            self.bert = AutoModel.from_config(config, attn_implementation="sdpa")
        else:
            self.bert = AutoModel.from_pretrained(base, attn_implementation="sdpa", reference_compile=False)
        cfg = self.bert.config
        self.L, self.window = cfg.num_hidden_layers, cfg.local_attention
        self.grad_ckpt = grad_ckpt
        self.lora_gate, self._qmask = None, None   # question-path LoRA; set by the trainer / runtime
        d = cfg.hidden_size
        self.rel = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_REL))
        self.loc_s, self.loc_q = nn.Linear(d, 256), nn.Linear(d, 256)
        # ev_s / ev_q: a bilinear evidence head that the MaxSim head replaced; unused, kept so checkpoints load
        self.ev_s, self.ev_q, self.ev_u = nn.Linear(d, 256), nn.Linear(d, 256), nn.Linear(d, 1)
        self.ms_s, self.ms_q = nn.Linear(d, 128), nn.Linear(d, 128)
        self.ms_scale = nn.Parameter(torch.tensor(10.0))
        self.chans = chans
        self.s_idx = [i for i, c in enumerate(chans) if c[1] == "s"]
        self.q_idx = [i for i, c in enumerate(chans) if c[1] == "q"]
        n_extra = len(self.s_idx) - 1   # state channel 0 is the MaxSim head above
        self.xs_s = nn.ModuleList(nn.Linear(d, 128) for _ in range(n_extra))
        self.xs_q = nn.ModuleList(nn.Linear(d, 128) for _ in range(n_extra))
        self.xs_u = nn.ModuleList(nn.Linear(d, 1) for _ in range(n_extra))
        self.xs_scale = nn.Parameter(torch.full((max(n_extra, 1),), 10.0))
        if self.q_idx:
            self.qh = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, len(self.q_idx)))

    def _layers(self, layers, h, pos, g, l):
        qm = self._qmask if self.lora_gate is not None else None

        def run(x, lay):
            if self.lora_gate is not None:   # set inside the call so checkpoint recomputation sees the same mask
                self.lora_gate.mask = qm
            return lay(x, attention_mask=g, sliding_window_mask=l, position_ids=pos)[0]

        for layer in layers:
            if self.grad_ckpt and self.training:
                h = checkpoint(run, h, layer, use_reentrant=False)
            else:
                h = run(h, layer)
        return h

    def forward(self, b):
        h = self.bert.embeddings(input_ids=b["ids"])
        if self.lora_gate is not None:
            self._qmask = (b["blk"] > 0)[..., None].to(h.dtype)
        g, l = masks(b["blk"], b["pos"], self.window)
        h = self.bert.final_norm(self._layers(self.bert.layers, h, b["pos"], g, l))
        flat = h.reshape(-1, h.shape[-1])
        qt = flat[b["q_tok"]]                                    # [Nq, Mq, d] this question's tokens
        qm = b["q_tok_m"]
        qpool = (qt * qm[..., None]).sum(1) / qm.sum(1, keepdim=True)
        state = h[b["q_pack"], : b["s_tok"].shape[1]]            # [Nq, Smax, d] (the pack's state, question-blind)
        return self._heads(qpool, state, qt, qm)

    def _heads(self, qpool, state, qt, qm):
        rel = self.rel(qpool)
        s = F.normalize(self.ms_s(state), dim=-1)
        q = F.normalize(self.ms_q(qt), dim=-1)
        sim = torch.einsum("nsk,nmk->nsm", s, q).masked_fill(qm[:, None] == 0, -2.0)
        ev = self.ms_scale * sim.max(-1).values + self.ev_u(state)[..., 0]
        out = {"q": qpool, "qt": qt, "qm": qm}
        loc = (self.loc_s(state) * self.loc_q(qpool)[:, None]).sum(-1) / 16
        out["loc"] = loc.float()
        ev = ev + loc
        out.update(rel=rel.float(), ev=ev.float())               # ev [Nq, Smax] over state token slots
        s_ch = [out["ev"]]
        for k in range(len(self.xs_s)):
            s = F.normalize(self.xs_s[k](state), dim=-1)
            q = F.normalize(self.xs_q[k](qt), dim=-1)
            sim = torch.einsum("nsk,nmk->nsm", s, q).masked_fill(qm[:, None] == 0, -2.0)
            s_ch.append((self.xs_scale[k] * sim.max(-1).values + self.xs_u[k](state)[..., 0] + out["loc"]).float())
        out["ev_s"] = torch.stack(s_ch, -1)                      # [Nq, Smax, Cs]
        if self.q_idx:
            out["ev_q"] = self.qh(qt).float()                    # [Nq, Mq, Cq]
        return out


class Answer(nn.Module):
    """Typed answer head, the hard stance bottleneck: level logit = tau[type] * log p(support | option block)."""

    def __init__(self, d=None, arm="hard"):
        super().__init__()
        if arm != "hard":
            raise ValueError("only the 'hard' answer arm is part of the release (docs/PORTING.md)")
        self.arm = arm
        self.tau = nn.Parameter(torch.ones(len(TYPES)))

    def typed_rel(self, q, rel):
        """Relation of typed option blocks (the shared relation head)."""
        return rel

    def levels(self, q, rel, typ):
        return self.tau[TYPES[typ]] * F.log_softmax(rel[:, :UNREL], -1)[:, SUP]


def unrel_logit(rel):
    """Question-block relation logits -> logit of p(unrelated) against the three related classes."""
    return rel[:, UNREL] - torch.logsumexp(rel[:, :UNREL], -1)
