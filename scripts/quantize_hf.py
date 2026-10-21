# SPDX-License-Identifier: MIT
"""Write Q8 / Q4 trunks next to an exported HF layout: <dir>/quantized/model-<scheme>.safetensors (horopter.quant).

    python scripts/quantize_hf.py export/horopter-1B --schemes w8 w4g32

With --verify (default on) the file is read back and every dequantized weight is checked bit for bit against the
rounding applied in place to the fp32 weights (the evaluated E41 arms), every other tensor against model.safetensors,
and the model is loaded once through horopter.runtime.load(..., quant=scheme) on CPU.
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horopter import quant as Q  # noqa: E402


def reference(w, scheme):
    """The E41 evaluation's in-place rounding (fp32 weights, symmetric absmax, round to nearest)."""
    bits, group = Q.SCHEMES[scheme]
    qmax = 2 ** (bits - 1) - 1
    w = w.float().clone()
    rows, cols = w.shape
    v = Q._groups(w, group)
    s = v.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return ((v / s).round().clamp(-qmax, qmax) * s).view(rows, -1)[:, :cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--schemes", nargs="+", default=["w8", "w4g32"], choices=list(Q.SCHEMES))
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    a = ap.parse_args()
    d = Path(a.dir)
    sd = load_file(str(d / "model.safetensors"))
    for scheme in a.schemes:
        out, meta = Q.pack(sd, scheme)
        f = d / Q.file_of(scheme)
        f.parent.mkdir(exist_ok=True)
        save_file(out, str(f), metadata=meta)
        n = sum(bool(Q.LINEAR.match(k)) for k in sd)
        print(f"{f}: {f.stat().st_size / 1e9:.3f} GB, {n} Linear weights quantized ({scheme})")
        if not a.verify:
            continue
        with safe_open(str(f), "pt") as h:
            back = Q.unpack(load_file(str(f)), h.metadata())
        assert back.keys() == sd.keys(), "key set differs from model.safetensors"
        for k, v in sd.items():
            if Q.LINEAR.match(k):
                assert torch.equal(back[k], reference(v, scheme)), f"{k}: dequantized weight != E41 rounding"
            else:
                assert back[k].dtype == v.dtype and torch.equal(back[k], v), f"{k}: changed"
        from horopter import runtime
        st = runtime.load(d, device="cpu", quant=scheme)
        print(f"  verified: bitwise equal to the E41 rounding; runtime.load ok ({len(st.bank)} adapters incl. base)")


if __name__ == "__main__":
    main()
