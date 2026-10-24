# SPDX-License-Identifier: MIT
"""Train one question-path LoRA adapter on a frozen exported trunk.

    python training/train_adapter.py --trunk export/horopter-0.4B --name nli --data data/v3/e24/expert_nli.jsonl \
        [--steps 2500] [--lr 3e-4] [--max_state 512]

Data: JSONL rows {id, source, state, questions: {qid: typed question}, gold: {qid: {"probabilities": {option: p}}}}.
Loss = typed CE (soft targets: the row's gold distribution) + typed stance (support t, refute 1 - t), as the trunk was
trained. Heads (relation, evidence, answer tau) train with the adapter. The last --dev_n rows are a dev split
(accuracy against the target argmax, reported for the trunk alone and with the adapter).
-> <out>/adapter.safetensors + adapter.json (+ train_log.json); <out> = --out or
   $HOROPTER_OUTPUTS/adapters/<trunk dir name>/<name>
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horopter import runtime  # noqa: E402
from horopter.losses import run_typed, typed_loss, typed_stance_loss  # noqa: E402
from horopter.packing import typed_pack  # noqa: E402


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def dev_acc(st, rows):
    hit = []
    for r in rows:
        out = st.answer(r["state"], r["questions"])
        for it in runtime.items_of(r):
            if it["gold"] is not None and it["qid"] in out:
                hit.append(int(np.argmax(out[it["qid"]]["p"]) == it["gold"]))
    return round(float(np.mean(hit)), 4) if hit else None, len(hit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--trunk", required=True, help="exported model dir (scripts/export_hf.py)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None, help="default: one pass over the training rows, 300..3000")
    ap.add_argument("--bs", type=int, default=8, help="rows per step")
    ap.add_argument("--lr", type=float, default=3e-4, help="LoRA lr")
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--typed_stance", type=float, default=1.0)
    ap.add_argument("--dev_n", type=int, default=300)
    ap.add_argument("--max_state", type=int, default=512, help="state window in tokens while training this adapter")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--description", default=None, help="adapter.json description: what its questions are about "
                                                         "(a router routes by it)")
    ap.add_argument("--meta", default=None, help="a JSON file merged into adapter.json (build_router_data.py writes "
                                                  "one: routing_layout and targets)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    trunk = Path(a.trunk)
    out_dir = Path(a.out) if a.out else Path(os.environ.get("HOROPTER_OUTPUTS", "outputs")) / "adapters" / trunk.name / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    st = runtime.load(trunk, adapters=False, abstain_head=False, max_state=a.max_state, pretrained_init=True)
    if st.r != a.r:
        raise SystemExit(f"--r {a.r}: the trunk's config sets LoRA rank {st.r}")
    rows = [json.loads(l) for l in open(a.data)]
    rows, dev = rows[:-a.dev_n], rows[-a.dev_n:]
    train = []
    for r in rows:
        its = [it for it in runtime.items_of(r) if it["target"] is not None]
        if its:
            train.append((r, its))
    steps = a.steps or min(3000, max(300, math.ceil(len(train) / a.bs)))
    p_acc = dev_acc(st, dev)
    lora, heads = st.trainable()
    print(f"[{a.name}] trunk {trunk.name} train rows {len(train)} dev {len(dev)} steps {steps}; trunk dev acc {p_acc}",
          flush=True)

    m, ans = st.model, st.ans
    groups = [{"params": heads, "lr": a.head_lr}, {"params": lora, "lr": a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    rng = random.Random(a.seed)
    order, step, losses, curve, t0 = [], 0, [], [], time.time()
    m.train()
    while step < steps:
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        batch = [train[order.pop()] for _ in range(min(a.bs, len(order)))]
        pg = [typed_pack(r["state"], its) for r, its in batch]
        res, _ = run_typed(m, ans, st.col, *zip(*pg), [its for _, its in batch], st.device)
        flat = [it for _, its in batch for it in its]
        loss = typed_loss(res, flat)
        if a.typed_stance:
            loss = loss + a.typed_stance * typed_stance_loss(res, flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        losses.append(loss.item())
        if step % 100 == 0 or step == steps:
            curve.append({"step": step, "loss": round(float(np.mean(losses)), 4), "sec": round(time.time() - t0)})
            losses = []
            print(f"[{a.name}] step {step}/{steps} loss {curve[-1]['loss']} {curve[-1]['sec']}s mem {peak_gb():.1f}G",
                  flush=True)
    train_sec = round(time.time() - t0)
    del opt
    a_acc = dev_acc(st, dev)
    sd = {k: v.float().cpu().contiguous() for k, v in st.adapter_state().items()}
    save_file(sd, str(out_dir / "adapter.safetensors"))
    meta = {"name": a.name, "trunk": trunk.name, "base": st.base, "p": st.run, "r": a.r, "kind": "loraq+heads",
            "data": Path(a.data).name, "train_rows": len(train), "steps": steps, "lr": a.lr, "head_lr": a.head_lr,
            "max_state": a.max_state, "train_sec": train_sec, "dev_acc_p": p_acc, "dev_acc": a_acc,
            "peak_mem_gb": round(peak_gb(), 1), "adapter_mb": round(sum(v.numel() for v in sd.values()) * 4 / 2**20, 1)}
    if a.description:
        meta["description"] = a.description
    if a.meta:
        meta.update(json.load(open(a.meta)))
    json.dump(meta, open(out_dir / "adapter.json", "w"), indent=1)
    json.dump(dict(meta, curve=curve), open(out_dir / "train_log.json", "w"), indent=1)
    print(f"[{a.name}] done {train_sec}s; dev acc trunk {p_acc} -> adapter {a_acc}", flush=True)


if __name__ == "__main__":
    main()
