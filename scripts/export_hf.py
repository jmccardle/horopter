# SPDX-License-Identifier: MIT
"""Convert a training checkpoint (+ adapters + abstention head) into the Horopter HF layout.

    python scripts/export_hf.py --ckpt outputs/e12/prod_e400m/model.pt --base jhu-clsp/ettin-encoder-400m \
        --max_state 2048 --adapter outputs/adapters/prod400m/check --abstain_head outputs/abstain_lr_T2.json \
        --name horopter-0.4b --out export/horopter-0.4B

Input:
    --ckpt          model.pt written by training/train_joint.py: {"model": trunk + heads state dict (bf16),
                    "answer": answer head}; a resumable stepNNNNNN.pt works too (its optimiser state is ignored)
    --adapter DIR   (repeatable) an adapter dir with adapter.safetensors + adapter.json (training/train_adapter.py,
                    training/train_evidence.py); --adapters_dir PARENT takes every <name>/ under PARENT
    --abstain_head  the head JSON written by training/fit_abstain_head.py
Output (--out):
    model.safetensors, config.json, tokenizer files, adapters/<name>/{adapter.safetensors, adapter.json},
    abstain_head.json
Tensors are written in the checkpoint's own dtypes (the trunk is bf16); horopter.runtime.load upcasts to fp32 exactly as
the research runtime did. With --verify (default on) the exported dir is loaded back on CPU and every adapter's keys
are checked against the model.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horopter.model import CHANS, N_REL, S_CH, TYPES  # noqa: E402
from horopter.runtime import FORMAT  # noqa: E402


def clean_meta(d):
    """Drop absolute paths from adapter / head metadata (keep the file name only)."""
    if isinstance(d, dict):
        return {k: clean_meta(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_meta(v) for v in d]
    if isinstance(d, str) and d.startswith("/"):
        return Path(d).name
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True, help="base encoder id, e.g. jhu-clsp/ettin-encoder-400m")
    ap.add_argument("--max_state", type=int, required=True, help="state window the trunk was trained with (prod: 2048)")
    ap.add_argument("--max_q", type=int, default=128)
    ap.add_argument("--lora_r", type=int, default=16, help="question-path LoRA rank of the adapters")
    ap.add_argument("--adapter", action="append", default=[], help="adapter dir (repeatable)")
    ap.add_argument("--adapters_dir", default=None, help="parent dir: every <name>/adapter.safetensors under it")
    ap.add_argument("--abstain_head", default=None)
    ap.add_argument("--name", default=None, help="model id, e.g. horopter-0.4b")
    ap.add_argument("--run", default=None, help="training run name (default: the checkpoint's directory name)")
    ap.add_argument("--release_date", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt)
    run = a.run or (ck.parent.name if ck.parent.name != "ckpt" else ck.parent.parent.name)

    sd = torch.load(ck, map_location="cpu", weights_only=False)
    tensors = {k: v.contiguous() for k, v in sd["model"].items()}
    tensors.update({f"answer.{k}": v.contiguous() for k, v in sd["answer"].items()})
    del sd
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    dtypes = sorted({str(v.dtype).replace("torch.", "") for v in tensors.values()})

    base_cfg = AutoConfig.from_pretrained(a.base)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(out)

    adapters = [Path(p) for p in a.adapter]
    if a.adapters_dir:
        adapters += sorted(p.parent for p in Path(a.adapters_dir).glob("*/adapter.safetensors"))
    names = []
    for d in adapters:
        name = d.name
        dst = out / "adapters" / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d / "adapter.safetensors", dst / "adapter.safetensors")
        meta = json.load(open(d / "adapter.json")) if (d / "adapter.json").exists() else {"name": name}
        json.dump(clean_meta(meta), open(dst / "adapter.json", "w"), indent=1)
        names.append(name)
    head = None
    if a.abstain_head:
        hj = json.load(open(a.abstain_head))
        json.dump(clean_meta(hj), open(out / "abstain_head.json", "w"), indent=1)
        head = "abstain_head.json"

    cfg = {
        "format": FORMAT,
        "architecture": "HoropterModel",
        "name": a.name or run,
        "run": run,
        "release_date": a.release_date,
        "base_model": a.base,
        "base_config": base_cfg.to_dict(),
        "hidden_size": base_cfg.hidden_size,
        "num_hidden_layers": base_cfg.num_hidden_layers,
        "local_attention": base_cfg.local_attention,
        "max_state": a.max_state,
        "max_q": a.max_q,
        "weights_dtype": dtypes,
        "design": {"name": "B", "interaction_layers": "all", "state": "question-blind, cacheable",
                   "cross_global": False, "attn_implementation": "sdpa"},
        "heads": {"relation": ["support", "refute", "undetermined", "unrelated"][:N_REL],
                  "evidence_head": "maxsim", "loc_head": True, "channels": [list(c[:3]) + [list(c[3])] for c in CHANS],
                  "state_channels": S_CH},
        "answer": {"arm": "hard", "types": TYPES, "score_mode": "levels"},
        "lora": {"r": a.lora_r, "alpha": 2 * a.lora_r, "gate": "question tokens", "targets": "every nn.Linear in "
                 "the trunk layers"},
        "adapters": names,
        "abstain_head": head,
    }
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    print(f"exported {ck} -> {out}: {len(tensors)} tensors ({dtypes}), adapters {names}, head {head}", flush=True)

    if a.verify:
        from horopter.runtime import load
        st = load(out, device="cpu")
        want = set(st.base_state)
        for n in names:
            got = set(load_file(str(out / "adapters" / n / "adapter.safetensors")))
            if not got <= want:
                raise SystemExit(f"adapter {n}: keys not in the model: {sorted(got - want)[:5]}")
        print(f"verified: loads on CPU, adapters {names} match the model", flush=True)


if __name__ == "__main__":
    main()
