# SPDX-License-Identifier: MIT
"""Evaluation helpers of the joint trainer: typed decisions on held-out rows, held-out losses, e-SNLI evidence roles.

Row schema (typed): {"id", "source", "state", "questions": {qid: typed question}, "gold": {qid: gold}} with gold
choice = option key, noul = bool, score = level. Optional teacher answers: a JSONL of {"id", "answers": {qid: answer}}
with answer {"noul": p} or {"probabilities": {option: p}}.
"""
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch

from horopter.losses import roles_loss, run_typed
from horopter.model import CHANS, S_CH, SUP, UNREL
from horopter.packing import batches, options, to, typed_pack

ROLE_OF = {0: "SUPPORT", 1: "CONFLICT", 2: "UNSUPPORTED"}


# ---------------------------------------------------------------------------------------------- typed items
def probs_of(q, ans):
    if ans is None:
        return None
    names = options(q)
    if q["type"] == "noul" or "noul" in ans:
        p = float(ans["noul"])
        v = [p, 1 - p]
    else:
        pr = ans["probabilities"]
        if all(n in pr for n in names):
            v = [pr[n] for n in names]
        elif q["type"] == "score" and all(c in pr for c in q["criteria"]):
            v = [pr[c] for c in q["criteria"]]
        else:
            return None
    v = np.clip(np.asarray(v, float), 0, None)
    return v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v))


def gold_index(q, g):
    if g is None:
        return None
    if q["type"] == "noul":
        return 0 if g else 1
    if q["type"] == "score":
        return int(round(float(g)))
    return options(q).index(g) if g in options(q) else None


def apply_T(p, T):
    l = np.log(np.clip(p, 1e-4, 1)) / T
    return np.exp(l - np.logaddexp.reduce(l))


def load_teacher(path):
    out = {}
    if path:
        for l in open(path):
            d = json.loads(l)
            if d.get("answers"):
                out[d["id"]] = d["answers"]
    return out


def items_of(rows, teachers):
    """Flatten rows -> per-question items with gold index and each teacher's (raw) probability vector."""
    items = []
    for r in rows:
        for qid, q in r["questions"].items():
            it = {"row": r["id"], "qid": qid, "q": q, "type": q["type"], "src": r["source"], "heldout": r.get("heldout"),
                  "gold": gold_index(q, (r.get("gold") or {}).get(qid)), "t": {}}
            for t, ans in teachers.items():
                a = ans.get(r["id"])
                p = probs_of(q, a.get(qid)) if a else None
                if p is not None:
                    it["t"][t] = p
            items.append(it)
    return items


def eval_set(rows, teacher_path, Ts):
    """Rows -> [(row, items)], the teacher ("jev") calibrated with the training temperatures."""
    items = items_of(rows, {"jev": load_teacher(teacher_path)})
    for it in items:
        if "jev" in it["t"]:
            it["t"]["jev"] = apply_T(it["t"]["jev"], Ts[f"jev/{it['type']}"])
    by = defaultdict(list)
    for it in items:
        by[it["row"]].append(it)
    return [(r, by[r["id"]]) for r in rows if by.get(r["id"])]


# ---------------------------------------------------------------------------------------------- metrics
def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            tot += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(tot / len(conf))


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    r = np.argsort(np.argsort(s)) + 1.0
    for v in np.unique(s):  # average ties
        m = s == v
        r[m] = r[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def avg_prec(y, s):
    o = np.argsort(-np.asarray(s))
    y = np.asarray(y, float)[o]
    if y.sum() == 0:
        return float("nan")
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def summarise(preds):
    def agg(ps):
        ps = [(it, p) for it, p in ps if it["gold"] is not None]
        if not ps:
            return None
        acc = np.array([p.argmax() == it["gold"] for it, p in ps])
        conf = np.array([p.max() for _, p in ps])
        jv = [(it, p) for it, p in ps if "jev" in it["t"]]
        r = {"n": len(ps), "acc": round(float(acc.mean()), 4), "ece": round(float(ece(conf, acc)), 4)}
        if jv:
            r["jev_acc"] = round(float(np.mean([it["t"]["jev"].argmax() == it["gold"] for it, _ in jv])), 4)
            r["agree_jev"] = round(float(np.mean([p.argmax() == it["t"]["jev"].argmax() for it, p in jv])), 4)
            r["kl_jev"] = round(float(np.mean([(it["t"]["jev"] * (np.log(it["t"]["jev"] + 1e-9) - np.log(p + 1e-9))).sum()
                                               for it, p in jv])), 4)
        return r

    groups = {
        "in_mixture": [x for x in preds if not x[0]["heldout"] and x[0]["src"] != "typed_decisions" and not x[0]["src"].startswith("x_")],
        "heldout_tasks": [x for x in preds if x[0]["heldout"]],
        "typed_decisions": [x for x in preds if x[0]["src"] == "typed_decisions"],
        "x_nli": [x for x in preds if x[0]["src"].startswith("x_")],
    }
    res = {k: agg(v) for k, v in groups.items()}
    by_src = defaultdict(list)
    for x in preds:
        by_src[x[0]["src"]].append(x)
    res["by_source"] = {s: agg(v) for s, v in sorted(by_src.items())}
    return res


def nll_of(preds):
    """Held-out answer loss of [(item, p)]: NLL of gold, and cross-entropy against the calibrated teacher."""
    g = [-math.log(float(p[it["gold"]]) + 1e-9) for it, p in preds if it["gold"] is not None]
    j = [-float((it["t"]["jev"] * np.log(p + 1e-9)).sum()) for it, p in preds if "jev" in it["t"]]
    return {"n_gold": len(g), "nll_gold": round(float(np.mean(g)), 4) if g else None,
            "n_jev": len(j), "ce_jev": round(float(np.mean(j)), 4) if j else None}


# ---------------------------------------------------------------------------------------------- predictions
@torch.no_grad()
def predict(model, ans, col, evset, max_blocks, device, bcfg, mismatch=False):
    """Per eval item: level probabilities, stance of each option block, question-level p(unrelated)."""
    model.eval()
    rows = evset
    if mismatch:   # deterministic: each row's questions over the state of the next row from another source
        rows = []
        for i, (r, its) in enumerate(evset):
            for k in range(1, len(evset)):
                d = evset[(i + k * 97) % len(evset)][0]
                if d["source"] != r["source"]:
                    rows.append((dict(r, state=d["state"]), its))
                    break
    out = []
    for batch in batches(rows, max_blocks, random.Random(0), **bcfg):
        pg = [typed_pack(r["state"], its, opts=not mismatch) for r, its in batch]
        items = [its for _, its in batch]
        res, ul = run_typed(model, ans, col, *zip(*pg), items, device)
        pu = torch.sigmoid(ul).cpu().numpy()
        flat = [it for its in items for it in its]
        if mismatch:
            out += [(it, None, None, float(u)) for it, u in zip(flat, pu)]
            continue
        for (z, rel), it, u in zip(res, flat, pu):
            p = z.float().softmax(-1)
            out.append((it, p.cpu().numpy(), rel[:, :UNREL].softmax(-1).cpu().numpy(), float(u)))
    model.train()
    return out


def typed_extras(P, Pm, ans):
    lv = [(it, p, st) for it, p, st, _ in P]
    return {
        "answer_eq_argmax_support": round(float(np.mean([p.argmax() == st[:, SUP].argmax() for _, p, st in lv])), 4),
        "mean_p_unrel": round(float(np.mean([u for *_, u in P])), 4),
        "mean_p_unrel_mismatched": round(float(np.mean([u for *_, u in Pm])), 4),
        "mismatch_auroc_p_unrel": round(auroc([0] * len(P) + [1] * len(Pm), [u for *_, u in P] + [u for *_, u in Pm]), 4),
        "tau": [round(float(v), 3) for v in ans.tau],
    }


def conflict_only(b):
    """Supervise only the refute (CONFLICT) state channel of a batch (the e-SNLI contradiction stream)."""
    keep = S_CH.index("CONFLICT")
    m = torch.zeros_like(b["es_m"])
    m[..., keep] = b["es_m"][..., keep]
    b["es_m"], b["eq_m"] = m, torch.zeros_like(b["eq_m"])
    return b


@torch.no_grad()
def roles_val(model, col_r, rows, ev_weight, loc_weight, device, roles_bs=8, conflict_bs=8):
    """Training losses r_rel / r_ev / c_rel / c_ev on e-SNLI roles test, computed as in training. Test rows carry 3
    annotators; annotator 0 stands in for the single training annotator. c_*: one contradiction per pack, CONFLICT
    state channel only, as the conflict stream."""
    model.eval()
    one = lambda ws: ws[0] if ws and isinstance(ws[0], list) else ws   # noqa: E731
    packs = [{"state": r["state"], "source": "esnli", "n_units": 1,
              "questions": [{**q, "src": "esnli", "words": one(q.get("pwords")), "hwords": one(q.get("hwords"))}
                            for q in r["questions"]]} for r in rows]
    conf = [{"state": p["state"], "n_units": 1, "questions": [{"text": q["text"], "label": 1, "words": q["words"],
                                                               "src": "esnli", "id": q.get("id")}]}
            for p in packs for q in p["questions"] if q["label"] == 1 and q["words"]]
    acc = defaultdict(list)
    for k, ps, bs in (("r", packs, roles_bs), ("c", conf, conflict_bs)):
        for i in range(0, len(ps), bs):
            b = col_r(ps[i:i + bs])
            if k == "c":
                b = conflict_only(b)
            b = to(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                out = model(b)
            _, l_r, l_e = roles_loss(out, b, CHANS, ev_weight, loc_weight)
            acc[f"{k}_rel"].append(l_r.item())
            acc[f"{k}_ev"].append(l_e.item())
    model.train()
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()} | {"n_roles_packs": len(packs), "n_conflict": len(conf)}


def _sorted_chunks(rows, bs):
    idx = sorted(range(len(rows)), key=lambda i: len(rows[i]["state"]) + sum(len(q["text"]) for q in rows[i]["questions"]))
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]


@torch.no_grad()
def predict_roles(model, col, packs, device, bs=32):
    """Per question: answer probs, per-word scores per state channel and per question channel."""
    model.eval()
    out = []
    for chunk in _sorted_chunks(packs, bs):
        b = col(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            o = model(to(b, device))
        P = o["rel"].softmax(-1).cpu().numpy()
        es = o["ev_s"].sigmoid().cpu().numpy()
        eq = o["ev_q"].sigmoid().cpu().numpy() if "ev_q" in o else None
        j = 0
        for p in chunk:
            for q in p["questions"]:
                t2w, nw = b["w_info"][j]
                ws = np.zeros((nw, es.shape[-1]))
                for t, w in enumerate(t2w):
                    if w >= 0:
                        ws[w] = np.maximum(ws[w], es[j, t])
                hs = None
                if eq is not None:
                    hs = np.zeros((len(q["text"].split()), eq.shape[-1]))
                    for t, w in enumerate(b["q_w"][j]):
                        if w >= 0:
                            hs[w] = np.maximum(hs[w], eq[j, t])
                out.append({"q": q, "pack": p, "p": P[j], "s": ws, "h": hs})
                j += 1
    model.train()
    return out


def score_role(items, get_scores, side):
    """Mean over (pair, annotator) of AP and F1@0.5; annotators with an empty highlight are skipped."""
    aps, f1s = [], []
    for x in items:
        sc = get_scores(x)
        if sc is None:
            continue
        for a in (x["q"]["pwords"] if side == "s" else x["q"]["hwords"]):
            g = np.zeros(len(sc), bool)
            g[[w for w in a if w < len(sc)]] = True
            if not g.any():
                continue
            aps.append(avg_prec(g, sc))
            pred = sc > 0.5
            tp = (pred & g).sum()
            f1s.append(0.0 if tp == 0 else 2 * tp / (pred.sum() + g.sum()))
    return {"ap": round(float(np.nanmean(aps)), 4), "f1": round(float(np.mean(f1s)), 4), "n": len(aps)} if aps else None


def role_eval(pr, chans):
    """Per-role evidence AP / F1 on e-SNLI test (both sides, 3 annotators), gold- and predicted-label-conditioned."""
    s_ch = [c for c in chans if c[1] == "s"]
    q_ch = [c for c in chans if c[1] == "q"]

    def chan(side, role):
        lst = s_ch if side == "s" else q_ch
        for k, c in enumerate(lst):
            if c[2] == "role" and c[0] in (role, "any"):
                return k
        return None

    res = {"gold_label": {}, "pred_label": {}, "rule_floor": {}}
    for lab, role in ROLE_OF.items():
        items = [x for x in pr if x["q"]["label"] == lab]
        for side in ("s", "q"):
            if lab == 2 and side == "s":
                continue                      # neutral premises carry no highlights
            key = f"{role}/{'prem' if side == 's' else 'hyp'}"
            k = chan(side, role)
            arr = "s" if side == "s" else "h"
            if k is not None:
                res["gold_label"][key] = score_role(items, lambda x: x[arr][:, k], side)

                def by_pred(x, side=side, arr=arr):   # channel picked by the model's own answer
                    kk = chan(side, ROLE_OF[int(np.argmax(x["p"][:3]))])
                    if kk is None:
                        return np.zeros(len(x["pack"]["state"].split() if side == "s" else x["q"]["text"].split()))
                    return x[arr][:, kk]
                res["pred_label"][key] = score_role(items, by_pred, side)
            if side == "q" and all("qcue" in x["q"] for x in items):
                res["rule_floor"][key] = score_role(items, lambda x: np.isin(np.arange(len(x["q"]["text"].split())), x["q"]["qcue"]["novel"]).astype(float), side)
    det = {}
    pair = lambda x, arr, k: float(x[arr][:, k].max()) if x[arr] is not None and len(x[arr]) else 0.0   # noqa: E731
    ks, kc, ku = chan("s", "SUPPORT"), chan("s", "CONFLICT"), chan("q", "UNSUPPORTED")
    if ks is not None and kc is not None and ks != kc:
        sub = [x for x in pr if x["q"]["label"] in (0, 1)]
        det["conflict_vs_support_state"] = round(auroc([x["q"]["label"] == 1 for x in sub],
                                                       [pair(x, "s", kc) - pair(x, "s", ks) for x in sub]), 4)
    if ku is not None and chan("q", "SUPPORT") != ku:
        sub = [x for x in pr if x["q"]["label"] in (0, 2)]
        det["unsupported_vs_support_hyp"] = round(auroc([x["q"]["label"] == 2 for x in sub],
                                                        [pair(x, "h", ku) for x in sub]), 4)
    res["role_detect_auroc"] = det
    res["acc3"] = round(float(np.mean([np.argmax(x["p"][:3]) == x["q"]["label"] for x in pr])), 4)
    return res
