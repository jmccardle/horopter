# SPDX-License-Identifier: MIT
"""Abstention head: extract the 21 answer / relation / evidence features on an exported trunk, then fit the logistic
head on the trunk's own errors (the E35 "lr_T2" head) and export it as abstain_head.json.

    # 1. features (one pass of the trunk over held-out sets with gold)
    python training/fit_abstain_head.py features --trunk export/horopter-0.4B --out feats/ \
        --set pool_eval=$HOROPTER_DATA/data/v3/e09/pool_eval.jsonl \
        --set a25_dev=$HOROPTER_DATA/data/v3/prod/eval_a25_dev.jsonl \
        --set a25_test=$HOROPTER_DATA/data/v3/prod/eval_a25_test.jsonl
    # 2. fit on pool_eval + a25_dev, check on a25_test (never fitted on)
    python training/fit_abstain_head.py fit --feats feats/ --out abstain_head.json

Rows: {id, source, state, questions, gold}. Per item: f = [answer: top1, top2, margin, normalised entropy, log n_opt,
is_noul, is_choice, is_score | relation: relevance, stance of the argmax option (sup, ref, und), max sup and max ref
over options, mean und | evidence (argmax option, state tokens): max/mean SUPPORT, max/mean CONFLICT, share > 0.5 of
each]; wrong = 1 if argmax != gold (-1 without gold); long = the state is longer than the window.
Head: P(wrong) = sigmoid(((f - mu) / sd) . w + b); served abstain.p = max(head, p_unrelated).
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from horopter import runtime  # noqa: E402
from horopter.abstain import FEATURES  # noqa: E402
from horopter.model import S_CH, UNREL, unrel_logit  # noqa: E402
from horopter.packing import batches, state_len, to, typed_pack  # noqa: E402
import joint_eval as JE  # noqa: E402

TYPES = ("noul", "choice", "score")


@torch.no_grad()
def extract(st, rows, max_blocks, bcfg):
    """rows: [(row, items)] -> list of per-item dicts (f, wrong, long, meta)."""
    model, ans, col = st.model, st.ans, st.col
    model.eval()
    s_sup, s_ref = S_CH.index("SUPPORT"), S_CH.index("CONFLICT")
    out = []
    for batch in batches(rows, max_blocks, random.Random(0), **bcfg):
        pg = [typed_pack(r["state"], its) for r, its in batch]
        packs, groups = zip(*pg)
        b = to(col(list(packs)), st.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=st.autocast):
            o = model(b)
        base = 0
        for (r, its), pack, grp in zip(batch, packs, groups):
            for g, it in zip(grp, its):
                qb = base + g["qb"]
                u = float(torch.sigmoid(unrel_logit(o["rel"][qb: qb + 1])[0]))
                sl = slice(base + g["opts"][0], base + g["opts"][-1] + 1)
                q = o["q"][sl]
                rel = ans.typed_rel(q, o["rel"][sl])
                p = ans.levels(q, rel, it["type"]).float().softmax(-1).cpu().numpy()
                stc = rel[:, :UNREL].float().softmax(-1).cpu().numpy()
                k = int(p.argmax())
                ps = np.sort(p)[::-1]
                ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))
                m = b["s_m"][base + g["opts"][k]].bool()
                ev = torch.sigmoid(o["ev_s"][base + g["opts"][k]].float())[m]          # [n_state_tok, Cs]
                es, er = ev[:, s_sup], ev[:, s_ref]
                f = [ps[0], ps[1] if len(ps) > 1 else 0.0, ps[0] - (ps[1] if len(ps) > 1 else 0.0), ent,
                     math.log(len(p)), *(float(it["type"] == t) for t in TYPES),
                     1 - u, *stc[k], stc[:, 0].max(), stc[:, 1].max(), stc[:, 2].mean(),
                     float(es.max()) if len(es) else 0.0, float(es.mean()) if len(es) else 0.0,
                     float(er.max()) if len(er) else 0.0, float(er.mean()) if len(er) else 0.0,
                     float((es > 0.5).float().mean()) if len(es) else 0.0,
                     float((er > 0.5).float().mean()) if len(er) else 0.0]
                g_ = it.get("gold")
                out.append({"f": np.asarray(f, np.float32), "plc": max(u, ent),
                            "wrong": -1 if g_ is None else int(k != g_),
                            "long": int(state_len(r) > bcfg["max_state"] - 2),
                            "meta": {"row": it["row"], "qid": it["qid"], "src": it["src"], "type": it["type"],
                                     "p": [round(float(x), 4) for x in p], "gold": g_}})
            base += len(pack["questions"])
    return out


def cmd_features(a):
    torch.backends.cuda.matmul.allow_tf32 = True
    st = runtime.load(a.trunk, adapters=False, abstain_head=False, max_state=a.max_state)
    if a.adapter:   # features of a served adapter instead of the trunk alone
        st.add_adapter(Path(a.adapter).name, a.adapter)
        st.use(Path(a.adapter).name)
    bcfg = {"max_tokens": a.max_tokens, "max_slots": a.max_slots, "max_state": a.max_state}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in a.set:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path)][: a.limit]
        items = JE.items_of(rows, {})
        by = defaultdict(list)
        for it in items:
            by[it["row"]].append(it)
        rows = [(r, by[r["id"]]) for r in rows if by.get(r["id"])]
        for r, _ in rows:
            r.setdefault("_len", (len(st.tok(r["state"], add_special_tokens=False)["input_ids"]), 0))
        res = extract(st, rows, a.max_blocks, bcfg)
        np.savez(out_dir / f"{name}.npz", f=np.stack([x["f"] for x in res]),
                 plc=np.array([x["plc"] for x in res], np.float32), wrong=np.array([x["wrong"] for x in res], np.int8),
                 long=np.array([x["long"] for x in res], np.int8))
        with open(out_dir / f"{name}.meta.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x["meta"]) + "\n")
        w = [x["wrong"] for x in res if x["wrong"] >= 0]
        print(f"[abstain] {name}: {len(rows)} rows, {len(res)} items, wrong rate {np.mean(w) if w else float('nan'):.3f}",
              flush=True)
    json.dump({"trunk": str(Path(a.trunk).name), "adapter": a.adapter and Path(a.adapter).name,
               "max_state": a.max_state, "sets": [s.split("=", 1)[0] for s in a.set], "f_names": FEATURES},
              open(out_dir / "info.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------------------ fit
class Head(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Linear(d_in, 1)

    def forward(self, x):
        return self.net(x)[:, 0]


def aurc(score, wrong):
    o = np.argsort(score, kind="stable")                      # most confident (lowest abstain) first
    risk = np.cumsum(wrong[o]) / np.arange(1, len(o) + 1)
    return float(risk.mean())


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable") + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def ece(p, y, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def load(d, name):
    z = np.load(Path(d) / f"{name}.npz")
    return {k: z[k] for k in z.files}


def cmd_fit(a):
    d = Path(a.feats)
    fit_sets = [load(d, n) for n in a.fit.split(",")]
    X = np.concatenate([s["f"][s["wrong"] >= 0] for s in fit_sets]).astype(np.float32)
    y = np.concatenate([s["wrong"][s["wrong"] >= 0] for s in fit_sets]).astype(np.float32)
    torch.manual_seed(0)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
    m = Head(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(a.epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            j = perm[i:i + 256]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    w = m.net.weight.detach().numpy()[0]
    b = float(m.net.bias.detach().numpy()[0])
    check = None
    if a.check and (d / f"{a.check}.npz").exists():
        te = load(d, a.check)
        head = lambda F: 1 / (1 + np.exp(-(((F - mu) / sd) @ w + b)))   # noqa: E731
        g = te["wrong"] >= 0
        s = np.maximum(head(te["f"][g].astype(np.float32)), te["f"][g][:, 8] * -1 + 1)
        wr = te["wrong"][g].astype(float)
        check = {"set": a.check, "aurc": round(aurc(s, wr), 4), "auroc": round(auroc(wr, s), 4), "ece": round(ece(s, wr), 4)}
    info = json.load(open(d / "info.json")) if (d / "info.json").exists() else {}
    out = {"name": a.name, "features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": b,
           "trunk": info.get("trunk"), "max_state": info.get("max_state"), "n_fit": int(len(y)),
           "served_form": "abstain = max(head, p_unrelated)", "check": check}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(check), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--trunk", required=True, help="exported model dir")
    f.add_argument("--adapter", default=None, help="optional adapter dir to extract features with")
    f.add_argument("--set", action="append", required=True, help="name=path of a held-out JSONL with gold (repeatable)")
    f.add_argument("--out", required=True)
    f.add_argument("--max_state", type=int, default=2048)
    f.add_argument("--max_blocks", type=int, default=192)
    f.add_argument("--max_tokens", type=int, default=16384)
    f.add_argument("--max_slots", type=int, default=49152)
    f.add_argument("--limit", type=int, default=None, help="smoke: rows per set")
    g = sub.add_parser("fit")
    g.add_argument("--feats", required=True)
    g.add_argument("--fit", default="pool_eval,a25_dev", help="feature sets to fit on")
    g.add_argument("--check", default="a25_test", help="feature set to check on (never fitted on)")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--name", default="lr_T2")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_features(a) if a.cmd == "features" else cmd_fit(a)


if __name__ == "__main__":
    main()
