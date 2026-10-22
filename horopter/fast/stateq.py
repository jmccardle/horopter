# SPDX-License-Identifier: MIT
"""State KV cache. Design B makes the state encoding question-blind and adapter-blind (state tokens attend only
the state, and the question-path LoRA is gated off on them), so a single-pack forward splits into
    state pass:     the frozen trunk over [CLS] state [SEP] alone; keeps every layer's post-rotary K, V and the final
                    normed state hidden states (LRU-cached by state token ids)
    question pass:  the question blocks alone, LoRA on every token, attending the cached state K, V plus their own
                    block (same design-B and sliding-window masks as horopter.model.masks)
and the heads read the cached state. Same function as HoropterModel.forward up to kernel rounding; one state encoding
then serves the router-chosen adapter, the evidence adapter and later requests about the same state.

    split = stateq.install(st)          # patches st.model.forward; multi-pack batches (the router) keep the full path
"""
import collections

import torch
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from ..model import LoRALinear


def _lin(m, x, lora):
    if isinstance(m, LoRALinear):
        return m(x) if lora else m.base(x)
    return m(x)


def _qkv(lay, x, pos, lora):
    at = lay.attn
    qkv = _lin(at.Wqkv, lay.attn_norm(x), lora).view(x.shape[0], -1, 3, at.num_heads, at.head_dim)
    cos, sin = at.rotary_emb(qkv, position_ids=pos)
    q, k, v = qkv.transpose(3, 1).unbind(dim=2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def _out(lay, x, a, lora):
    at = lay.attn
    x = x + _lin(at.Wo, a.transpose(1, 2).reshape(x.shape[0], -1, at.all_head_size), lora)
    mlp = lay.mlp
    i, g = _lin(mlp.Wi, lay.mlp_norm(x), lora).chunk(2, dim=-1)
    return x + _lin(mlp.Wo, mlp.act(i) * g, lora)


def state_layer(lay, x, pos, band, gmask=None):
    """One frozen layer over state tokens only. band: the sliding-window mask of local layers; gmask: the mask of
    global layers (None = no padding, attend everything)."""
    q, k, v = _qkv(lay, x, pos, False)
    m = band if lay.attn.local_attention != (-1, -1) else gmask
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    return _out(lay, x, a, False), k, v


def question_layer(lay, x, pos, ks, vs, gm, lm):
    """One layer over question tokens, keys/values = cached state + the question tokens."""
    q, k, v = _qkv(lay, x, pos, True)
    m = lm if lay.attn.local_attention != (-1, -1) else gm
    a = F.scaled_dot_product_attention(q, torch.cat([ks, k], 2), torch.cat([vs, v], 2), attn_mask=m)
    return _out(lay, x, a, True)


class Split:
    def __init__(self, model, cache_size=4, compile=False):
        self.m, self.cache, self.cache_size = model, collections.OrderedDict(), cache_size
        self.full = model.forward
        self.hits = self.misses = 0
        self.sl, self.ql = state_layer, question_layer
        if compile:
            self.sl, self.ql = torch.compile(state_layer, dynamic=True), torch.compile(question_layer, dynamic=True)

    def reset(self):
        self.cache.clear()

    def encode_state(self, ids, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        m, w = self.m, self.m.window
        S = ids.shape[1]
        pos = torch.arange(S, device=ids.device)[None]
        band = ((pos[0, :, None] - pos[0, None, :]).abs() <= w // 2)[None, None]
        h = m.bert.embeddings(input_ids=ids)
        kv = []
        for lay in m.bert.layers:
            h, k, v = self.sl(lay, h, pos, band)
            kv.append((k, v))
        ent = (kv, m.bert.final_norm(h))
        self.cache[key] = ent
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return ent

    def forward(self, b):
        if b["ids"].shape[0] != 1:
            return self.full(b)
        blk, pos, ids = b["blk"][0], b["pos"][0], b["ids"]
        S = b["s_tok"].shape[1]                   # Smax = this pack's state length ([CLS] state [SEP])
        key = b.get("_state_key")
        if key is None:
            key = tuple(ids[0, :S].tolist())
        kv, hs = self.encode_state(ids[:, :S], key)
        # question tokens = every non-pad token after the state
        qsel = (blk[S:] > 0).nonzero()[:, 0] + S
        qb, qp = blk[qsel], pos[qsel]
        w = self.m.window
        same = qb[:, None] == qb[None, :]
        near_q = (qp[:, None] - qp[None, :]).abs() <= w // 2
        near_s = (qp[:, None] - torch.arange(S, device=qp.device)[None]).abs() <= w // 2
        to_s = torch.ones(len(qsel), S, dtype=torch.bool, device=qp.device)
        gm = torch.cat([to_s, same], 1)[None, None]
        lm = torch.cat([near_s, same & near_q], 1)[None, None]
        x = self.m.bert.embeddings(input_ids=ids[:, qsel])
        if self.m.lora_gate is not None:
            self.m.lora_gate.mask = None          # every token here is a question token
        for lay, (ks, vs) in zip(self.m.bert.layers, kv):
            x = self.ql(lay, x, qp[None], ks, vs, gm, lm)
        hq = self.m.bert.final_norm(x)[0]
        # map the flat [B * T] question-token indices of the batch to rows of hq
        T = ids.shape[1]
        row = torch.full((T,), 0, dtype=torch.long, device=ids.device)
        row[qsel] = torch.arange(len(qsel), device=ids.device)
        qt = hq[row[b["q_tok"] % T]]
        qm = b["q_tok_m"]
        qpool = (qt * qm[..., None]).sum(1) / qm.sum(1, keepdim=True)
        state = hs.expand(qt.shape[0], -1, -1)
        return self.m._heads(qpool, state, qt, qm)


def install(st, cache_size=4, compile=False):
    sp = Split(st.model, cache_size, compile)
    st.model.forward = sp.forward
    return sp
