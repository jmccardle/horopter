# SPDX-License-Identifier: MIT
"""Weight-only quantized trunks: Q8 (int8, one scale per output row) and Q4 (int4, one scale per 32 or 128 inputs).

Every Linear in the trunk (attention Wqkv / Wo, MLP Wi / Wo) is stored as integers plus fp32 scales: symmetric absmax,
round to nearest, no calibration. Embeddings, norms, heads, the answer head and the LoRA adapters stay as in
model.safetensors. horopter.runtime.load(path, quant="w8") dequantizes at load time, so the numbers are exactly those of
the quantized grid (the E41 evaluation), computed in fp32 / bf16; the file is the smaller download, not a faster kernel.

File: quantized/model-<scheme>.safetensors, metadata {"format": QFORMAT, "scheme", "bits", "group"}. For each quantized
weight key K: K.qweight (w8: int8 [rows, cols]; w4: uint8 [rows, padded cols / 2], two values per byte, low nibble first,
stored as q + 8), K.scales (fp32 [rows, groups]) and K.cols (int64 [1], the unpadded input width). Short last groups (cols not a multiple of the group) are zero-padded.
"""
import re

import torch

QFORMAT = "horopter-quant-1"
SCHEMES = {"w8": (8, 0), "w4g32": (4, 32), "w4g128": (4, 128)}
LINEAR = re.compile(r"^bert\.layers\.\d+\.(attn\.Wqkv|attn\.Wo|mlp\.Wi|mlp\.Wo)\.weight$")


def file_of(scheme):
    return f"quantized/model-{scheme}.safetensors"


def _groups(w, group):
    rows, cols = w.shape
    pad = (-cols) % group if group else 0
    return torch.nn.functional.pad(w, (0, pad)).view(rows, -1, group) if group else w.view(rows, 1, -1)


@torch.no_grad()
def quantize(w, scheme):
    """fp weight [rows, cols] -> (integer tensor [rows, groups, group], fp32 scales [rows, groups])."""
    bits, group = SCHEMES[scheme]
    qmax = 2 ** (bits - 1) - 1
    v = _groups(w.float(), group)
    s = v.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return (v / s).round().clamp(-qmax, qmax).to(torch.int8), s.squeeze(-1)


def dequantize(q, s, cols):
    rows = q.shape[0]
    return (q.float() * s.unsqueeze(-1)).view(rows, -1)[:, :cols]


def pack(sd, scheme):
    """model.safetensors dict -> quantized dict (+ metadata)."""
    bits, group = SCHEMES[scheme]
    out = {}
    for k, v in sd.items():
        if not LINEAR.match(k):
            out[k] = v
            continue
        q, s = quantize(v, scheme)
        q = q.view(q.shape[0], -1)
        if bits == 4:
            u = (q + 8).to(torch.uint8)
            q = u[:, 0::2] | (u[:, 1::2] << 4)
        out[f"{k}.qweight"], out[f"{k}.scales"] = q.contiguous(), s.contiguous()
        out[f"{k}.cols"] = torch.tensor([v.shape[1]], dtype=torch.int64)
    return out, {"format": QFORMAT, "scheme": scheme, "bits": str(bits), "group": str(group)}


def unpack(qsd, meta):
    """Quantized dict -> fp32 weights for the quantized keys, everything else as stored."""
    if meta.get("format") != QFORMAT:
        raise ValueError(f"quantized file format {meta.get('format')!r}, expected {QFORMAT!r}")
    bits, group = int(meta["bits"]), int(meta["group"])
    out = {}
    for k, v in qsd.items():
        if k.endswith(".scales") or k.endswith(".cols"):
            continue
        if not k.endswith(".qweight"):
            out[k] = v
            continue
        key = k[: -len(".qweight")]
        rows, cols = v.shape[0], int(qsd[f"{key}.cols"][0])
        if bits == 4:
            q = torch.stack([(v & 15).to(torch.int8) - 8, (v >> 4).to(torch.int8) - 8], -1).view(rows, -1)
        else:
            q = v
        s = qsd[f"{key}.scales"]
        out[key] = dequantize(q.view(rows, s.shape[1], -1), s, cols)
    return out
