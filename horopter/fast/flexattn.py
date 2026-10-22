# SPDX-License-Identifier: MIT
"""Block-sparse FlexAttention for the STATE pass (masked SDPA costs 3-3.5x the unmasked kernel on a Jetson Orin).

    local layers   banded block mask (|q - k| <= window // 2), block size BLOCK (128 default; try 64)
    global layers  no block mask (dense); only padding is excluded
Padding (graphq pads the state to a multiple of 64): keys k >= S_real are excluded, a pad row attends itself (so no
row is ever empty and no NaN reaches the cached K/V). Real rows therefore see exactly the keys graphq's masks give
them, so real-token hidden states and cached K/V equal the masked-SDPA path up to kernel rounding. Pad rows differ
(they also see real keys in band) but are never read: the question pass masks every pad key.

S_real is read INSIDE the call from the mask graphq already passes (gmask[0,0,:,0].sum(): column 0 of the global mask
is True exactly for the real rows) and copied into a static device scalar the score_mod loads, so one compiled kernel
serves every S_real of a bucket and the call is CUDA-graph safe. graphq.GSplit(flex=True) uses it as its state layer.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


BLOCK = 128                 # block-mask block size (flex_optime tries 64 as well)
WINDOW = 128                # Ettin-400m local window; |q - k| <= WINDOW // 2
_flex = {}                  # kernel-option tuple -> torch.compile(flex_attention); built lazily
_bm = {}                    # (Sp, window, block, device) -> BlockMask
_SR = {}                    # device -> int32 scalar buffer holding S_real


def compiled(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _flex:
        import torch._dynamo.config as dc
        for n in ("recompile_limit", "cache_size_limit"):   # one specialisation per (Sp, block mask); default 8 silently falls back to the unfused path
            if hasattr(dc, n):
                setattr(dc, n, 1024)
        _flex[key] = torch.compile(flex_attention, dynamic=False)
    return _flex[key]


def _sr(dev):
    if dev not in _SR:
        _SR[dev] = torch.zeros((), dtype=torch.int32, device=dev)
    return _SR[dev]


def band_block_mask(Sp, window, dev, block=None):
    block = block or BLOCK
    key = (Sp, window, block, str(dev))
    if key not in _bm:
        half = window // 2

        def band(b, h, q, k):
            return (q - k).abs() <= half
        _bm[key] = create_block_mask(band, None, None, Sp, Sp, device=dev, BLOCK_SIZE=block)
    return _bm[key]


def make_score_mod(sr):
    def pad_mod(score, b, h, q, k):
        return torch.where((k < sr) | (q == k), score, -float("inf"))
    return pad_mod


_MODS = {}


def _mod(dev):
    if dev not in _MODS:
        _MODS[dev] = make_score_mod(_sr(dev))
    return _MODS[dev]


def state_attn(q, k, v, S_real, window, is_local, block=None, flex=None):
    """q, k, v [1, H, Sp, hd]; S_real: int or 0-d tensor (device). Returns [1, H, Sp, hd]."""
    dev = q.device
    Sp = q.shape[2]
    sr = _sr(dev)
    if isinstance(S_real, torch.Tensor):
        sr.copy_(S_real)
    else:
        sr.fill_(int(S_real))
    bm = band_block_mask(Sp, window, dev, block) if is_local else None
    block = block or BLOCK
    ko = {} if block >= 128 else {"BLOCK_M": block, "BLOCK_N": block}   # tiles must divide the mask block size
    f = flex or compiled()
    return f(q, k, v, score_mod=_mod(dev), block_mask=bm, **({"kernel_options": ko} if ko else {}))


def state_layer_flex(lay, x, pos, band, gmask=None):
    """Same signature as stateq.state_layer. band is ignored (the block mask is the band); gmask only supplies S_real
    (None = no padding)."""
    from . import stateq
    q, k, v = stateq._qkv(lay, x, pos, False)
    Sp = x.shape[1]
    S_real = Sp if gmask is None else gmask[0, 0, :, 0].sum().to(torch.int32)
    is_local = lay.attn.local_attention != (-1, -1)
    a = state_attn(q, k, v, S_real, lay.attn.local_attention[1] * 2 if is_local else 0, is_local)
    return stateq._out(lay, x, a, False), k, v
