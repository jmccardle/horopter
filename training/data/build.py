# SPDX-License-Identifier: MIT
"""E34: training data for the 1B production run (CPU only; cached teacher answers only; re-runnable).

    HOROPTER_DATA=<data root> python training/data/lengths.py   # once per pool change
    HOROPTER_DATA=<data root> python training/data/build.py [--caps training/data/caps.json]

This is the recipe script of the beta (and, with build_prod2.py on top, beta-2) training data. It reads the typed
pools and cached teacher answers (Jev, decider, clef-flash) under $HOROPTER_DATA (see common.py); none ship with the
repo. The stacking report goes to $HOROPTER_OUTPUTS/e34/stack.json (default outputs/e34).

Steps (PLAN.md has the design and the schema):
  split     held-out long-state eval from a25, stratified by source, whole document groups, decontaminated against
            every training state by word 13-grams -> data/v3/prod/eval_a25_{dev,test}.jsonl (full rows)
  calibrate per teacher x question type temperature (E09's grid fit) on the 10% 'fit' bucket of the training rows
  stack     A2.2 per-cluster teacher weights + a pool temperature, linear and log-linear pools, fitted by NLL on the
            fit bucket, each cluster shrunk toward its parent; scored on the 5% 'dev' bucket -> results/stack.json
  targets   per question: 0.5 gold + 0.5 stacked committee where gold exists, else the stacked committee (renormalised
            over the teachers that answered); typed stance targets with the undetermined split; abstention target;
            in-window flag; generic-question flag -> data/v3/prod/targets/<pool>.jsonl.gz (ids only, no text)
  evidence  a25 gold evidence (char offsets) -> 3-role option-block labels + state word indices
            -> data/v3/prod/evidence_a25.jsonl.gz
  select    per-source decision caps for the typed stream (keep flag in the target files) -> manifest.json
"""
import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (A25_FAMILY, E09, POOL_PATH, POOLS, PREFERENCE_LONG, PROD, ROOT, TEACHERS, WINDOW, chain,  # noqa: E402,F401
                    cluster, family, fit_bucket, gold_index, group_key, h01, instr_of, options, probs_of)

RES = Path(os.environ.get("HOROPTER_OUTPUTS", "outputs")) / "e34"
GRID = np.exp(np.linspace(np.log(0.25), np.log(8), 81))          # E08 fit_temperature grid
SHRINK = 50.0                                                     # pseudo-items pulling a cluster toward its parent
MIN_FIT = 300                                                     # fewer fit items -> the cluster uses its parent's fit
GENERIC_STATES = 20                                               # an instruction asked of >= this many states
EVAL_DEV, EVAL_TEST = 100, 160                                    # rows per source (at most 10% / 15% of the source)
EVAL_LONG_SHARE = 0.5                                             # of each source quota, from states over the window
OVERLAP_MAX = 0.10                                                # eval rows with more 13-gram overlap are dropped
TARGET_ARM = "guard"                                              # linear pool + temperature + decider guard
# Sources whose train split decider-2b was trained on (E09: pool_train is decider's mixture; "decider wins only on its
# own training mixture"): every pool_train source but typed_decisions, their a23 copies, the NLI reformulations of
# SNLI / MNLI / FEVER, and the E33 rebuilds of mixture data (unclipped preference rows, ERASER BoolQ / MultiRC, RACE
# articles, MS MARCO v1.1, SQuAD paragraphs = QNLI's source). Filled in main() from pool_train's source list.
DTR = {"nli_snli", "nli_mnli", "nli_fever", "shp_long", "ultrafeedback_long", "hh_rlhf_long", "helpsteer2_long",
       "helpsteer3_long", "boolq_doc", "multirc_doc", "race_article", "msmarco_multi", "squad_doc"}
T0 = time.time()


def log(*a):
    print(f"[e34 {time.time() - T0:6.0f}s]", *a, flush=True)


def jl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)


def load_lengths(pool):
    d = json.load(gzip.open(PROD / "lengths" / f"{pool}.json.gz", "rt"))
    return {i: (s, q, b, c) for i, s, q, b, c in zip(d["ids"], d["s"], d["q"], d["b"], d["cut2048"])}


def load_teacher(name, pool):
    out, p = {}, E09 / "raw" / f"{name}.{pool}.jsonl"
    if p.exists():
        for d in jl(p):
            if d.get("answers"):
                out[d["id"]] = d["answers"]
    return out


# ------------------------------------------------------------------------------------------------ window validity

def response_span(state, k):
    a = state.find(f"\n\nResponse {k}:\n")
    if a < 0:
        return None
    b = state.find(f"\n\nResponse {k + 1}:\n", a + 1)
    return a, (b if b >= 0 else len(state))


def in_window(r, qid, ntok, cut):
    """Is this question's label decidable from the first WINDOW - 2 state tokens? (True for every state that fits.)"""
    if ntok <= WINDOW - 2:
        return True
    src, g = r["source"], r["gold"].get(qid)
    ev = (r.get("evidence") or {}).get(qid)
    if ev:                                            # every gold evidence span inside the window
        return max(e for _, e in ev) <= cut
    if g is None:
        return False                                  # teacher-only label read from the whole state
    if src == "contractnli_doc":                      # NotMentioned in the document -> also in any prefix
        return True
    if src == "qasper_doc" and g is False:            # unanswerable for every annotator -> also in any prefix
        return True
    if src == "wice_doc" and g == "not_supported":    # a prefix supports no more than the whole text
        return True
    if src == "ragtruth_multi" and g is True:         # the reference precedes the responses; response k must fit
        m = re.search(r"Response (\d+)", r["questions"][qid]["instructions"])
        sp = response_span(r["state"], int(m.group(1))) if m else None
        return bool(sp) and sp[1] <= cut
    return False


# ------------------------------------------------------------------------------------------------ eval split (a25)

def grams(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def make_split(a25, L25, train_states):
    """-> {row id: 'dev' | 'test'}; whole groups, per-source quotas, half from states over the window."""
    by_src = defaultdict(lambda: defaultdict(list))
    for r in a25:
        if any(v is not None for v in r["gold"].values()):
            by_src[r["source"]][group_key(r)].append(r)
    n_src = Counter(r["source"] for r in a25)
    # 13-gram index of every training state outside a25 (E09 / E14 pools); a25 groups are disjoint by construction
    log("eval split: 13-gram index of the E09/E14 training states")
    idx = set()
    for s in train_states:
        idx |= grams(s)
    log(f"eval split: {len(idx):,} training 13-grams")
    split, stats = {}, {}
    for src in sorted(by_src):
        groups = sorted(by_src[src].items(), key=lambda kv: h01(kv[0], "e34eval"))
        quota = {"dev": min(EVAL_DEV, int(0.10 * n_src[src])), "test": min(EVAL_TEST, int(0.15 * n_src[src]))}
        need = {(s, lg): math.ceil(quota[s] * (EVAL_LONG_SHARE if lg else 1 - EVAL_LONG_SHARE))
                for s in quota for lg in (True, False)}
        got, dropped, used = Counter(), 0, set()
        # pass 1: long groups fill the long half of each quota, short groups the short half; pass 2: any group fills
        # what is left (sources with few long states)
        for pas in (1, 2):
            for g, rows in groups:
                if g in used:
                    continue
                if all(got[s, True] + got[s, False] >= quota[s] for s in quota):
                    break
                longs = any(L25[r["id"]][0] > WINDOW - 2 for r in rows)
                if pas == 1:
                    sp = next((s for s in ("test", "dev") if got[s, longs] < need[s, longs]), None)
                else:
                    sp = next((s for s in ("test", "dev") if got[s, True] + got[s, False] < quota[s]), None)
                if sp is None:
                    continue
                used.add(g)
                ov = max(len(gr & idx) / max(1, len(gr)) for gr in (grams(r["state"]) for r in rows))
                if ov > OVERLAP_MAX:
                    dropped += 1
                    continue
                for r in rows:
                    split[r["id"]] = sp
                got[sp, longs] += len(rows)
        stats[src] = {"rows": n_src[src], "dev": got["dev", True] + got["dev", False],
                      "test": got["test", True] + got["test", False],
                      "dev_long_rows": got["dev", True], "test_long_rows": got["test", True],
                      "groups_dropped_overlap": dropped}
    return split, stats


# ------------------------------------------------------------------------------------------------ calibration / stacking

def fit_T(L, y):
    """E08 fit_temperature, vectorised: L list of clipped log-prob arrays, y gold indices."""
    best, bi = None, 0
    by_k = defaultdict(list)
    for l, g in zip(L, y):
        by_k[len(l)].append((l, g))
    arrs = {k: (np.stack([l for l, _ in v]), np.array([g for _, g in v])) for k, v in by_k.items()}
    n = len(L)
    for i, T in enumerate(GRID):
        tot = 0.0
        for X, g in arrs.values():
            Z = X / T
            lse = np.logaddexp.reduce(Z, axis=1)
            tot += float((lse - Z[np.arange(len(g)), g]).sum())
        if best is None or tot / n < best:
            best, bi = tot / n, i
    return float(GRID[bi])


def apply_T(p, T):
    l = np.log(np.clip(p, 1e-4, 1)) / T
    return np.exp(l - np.logaddexp.reduce(l))


DEC = TEACHERS.index("decider")


def pack(items):
    """items -> padded tensors: logP [N,T,K] (calibrated log-probs; 0 where missing), avail [N,T], valid [N,K], y [N],
    pin [N] (decider trained on this source: its weight is pinned, see decider_trained)."""
    K = max(len(it["opts"]) for it in items)
    N, nT = len(items), len(TEACHERS)
    lp = np.zeros((N, nT, K), np.float32)
    av = np.zeros((N, nT), np.float32)
    va = np.zeros((N, K), bool)
    y = np.zeros(N, np.int64)
    pin = np.zeros(N, bool)
    for i, it in enumerate(items):
        k = len(it["opts"])
        va[i, :k] = True
        y[i] = it["gold"] if it["gold"] is not None else 0
        pin[i] = it["dtr"]
        for j, t in enumerate(TEACHERS):
            if t in it["p"]:
                lp[i, j, :k] = np.log(np.clip(it["p"][t], 1e-6, 1))
                av[i, j] = 1
    return tuple(map(torch.tensor, (lp, av, va, y, pin)))


def pool_weights(a, av, pin, guard):
    """Linear-pool weights per item, renormalised over the teachers that answered. guard: on items whose source decider
    was trained on, decider's share is capped at its plain-mean share (1 / n answering) and the others split the rest by
    their fitted relative weights (a proper score against gold would otherwise reward decider's memorised training
    labels). A cap, not a pin (2026-10-16): pinning raised decider to 1/3 on long documents of its training sources
    (squad_doc, boolq_doc, race_article, *_long) where it is weak, long_reading dev NLL 0.137 -> 0.187."""
    w = torch.softmax(a, 0)[None] * av
    if guard and pin.any():
        o = w.clone()
        o[:, DEC] = 0
        os_ = o.sum(1, keepdim=True)
        n = av.sum(1, keepdim=True).clamp(min=1)
        hd = av[:, DEC:DEC + 1] > 0
        fitted = w[:, DEC:DEC + 1] / w.sum(1, keepdim=True).clamp(min=1e-12)
        share = torch.where(hd, torch.minimum(fitted, 1.0 / n), torch.zeros_like(n))
        wp = torch.where(os_ > 0, o / os_.clamp(min=1e-12) * (1 - share), o)
        wp[:, DEC] = torch.where(os_[:, 0] > 0, share[:, 0], hd[:, 0].float())
        w = torch.where(pin[:, None], wp, w)
    return w / w.sum(1, keepdim=True).clamp(min=1e-12)


def pool_logp(theta, lp, av, va, pin, kind, guard=False):
    """theta = [3 teacher logits, log pool temperature] -> log p [N,K] of the pooled, sharpened forecast."""
    a, logT = theta[:-1], theta[-1]
    neg = torch.finfo(lp.dtype).min / 4
    if kind == "lin":   # linear pool, then a temperature
        w = pool_weights(a, av, pin, guard)
        m = torch.logsumexp(lp + torch.log(w.clamp(min=1e-12))[..., None], 1)   # [N,K]
    else:               # log-linear pool, exponents rescaled so the available ones keep the full total (report only)
        b = torch.nn.functional.softplus(a)[None] * av
        b = b * (torch.nn.functional.softplus(a).sum() / b.sum(1, keepdim=True).clamp(min=1e-12))
        m = (b[..., None] * lp).sum(1)
    z = (m / torch.exp(logT)).masked_fill(~va, neg)
    return z - torch.logsumexp(z, 1, keepdim=True)


ARMS = {"lin": ("lin", False), "log": ("log", False), "guard": ("lin", True)}   # name -> (pool, decider guard)


def fit_pool(data, arm, prior):
    lp, av, va, y, pin = data
    kind, guard = ARMS[arm]
    th = prior.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([th], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
    lam = SHRINK / len(y)

    def closure():
        opt.zero_grad()
        nll = -pool_logp(th, lp, av, va, pin, kind, guard).gather(1, y[:, None]).mean()
        loss = nll + lam * ((th - prior) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return th.detach()


def predict(theta, data, arm):
    kind, guard = ARMS[arm]
    with torch.no_grad():
        return pool_logp(theta, *data[:3], data[4], kind, guard)


def nll_of(theta, data, arm):
    return float(-predict(theta, data, arm).gather(1, data[3][:, None]).mean())


def mean_logp(data):
    lp, av, va = data[:3]
    w = av / av.sum(1, keepdim=True).clamp(min=1e-12)
    m = torch.logsumexp(lp + torch.log(w.clamp(min=1e-12))[..., None], 1).masked_fill(~va, -1e30)
    return m - torch.logsumexp(m, 1, keepdim=True)


def nll_table(items, th):
    """Dev NLL per cluster: each teacher, plain mean, stacked (linear, log-linear, linear + decider guard = the target),
    on items with all three teachers ('common') and on all items (missing teachers renormalised). Rows split by whether
    decider trained on the source ('in_decider_mix' / 'out_of_decider_mix')."""
    by = defaultdict(list)
    for it in items:
        by[it["cl"]].append(it)
        by["all"].append(it)
        by["all/" + ("in_decider_mix" if it["dtr"] else "out_of_decider_mix")].append(it)
    out = {}
    for cl, its in sorted(by.items()):
        row = {}
        for sub in ("common", "all"):
            s = [it for it in its if sub == "all" or all(t in it["p"] for t in TEACHERS)]
            if len(s) < 50:
                continue
            d = pack(s)
            r = {"n": len(s)}
            for j, t in enumerate(TEACHERS):
                has = d[1][:, j] > 0
                if has.sum() >= 50:
                    lp = d[0][has, j].masked_fill(~d[2][has], -1e30)
                    lp = lp - torch.logsumexp(lp, 1, keepdim=True)
                    r[t] = round(float(-lp.gather(1, d[3][has][:, None]).mean()), 4)
                    if sub == "all":
                        r[f"n_{t}"] = int(has.sum())
            r["mean"] = round(float(-mean_logp(d).gather(1, d[3][:, None]).mean()), 4)
            for arm in ARMS:   # every item scored with its own cluster's parameters
                v = 0.0
                for c2 in sorted({it["cl"] for it in s}):
                    s2 = [it for it in s if it["cl"] == c2]
                    v += nll_of(th[arm][c2], pack(s2), arm) * len(s2)
                r[f"stack_{arm}"] = round(v / len(s), 4)
            row[sub] = r
        out[cl] = row
    return out


# ------------------------------------------------------------------------------------------------ evidence (a25)

def word_spans(text):
    sp, i = [], 0
    for w in text.split():
        s = text.index(w, i)
        sp.append((s, s + len(w)))
        i = s + len(w)
    return sp


def words_in(spans, ws):
    """Whitespace-word indices overlapping any char span (the Collator's state word map)."""
    import bisect
    starts = [a for a, _ in ws]
    out = set()
    for s, e in spans:
        k = max(0, bisect.bisect_right(starts, s) - 1)
        while k < len(ws) and ws[k][0] < e:
            if ws[k][1] > s:
                out.add(k)
            k += 1
    return sorted(out)


SUP, REF, SKIP = 0, 1, -1


def evidence_labels(r, qid):
    """Per option block of a question with gold evidence: SUPPORT (0) for the gold option, CONFLICT (1) for options
    the evidence excludes, -1 unsupervised. None when the evidence carries no state-side role."""
    q, g = r["questions"][qid], r["gold"].get(qid)
    if g is None:
        return None
    opts = options(q)
    gi = gold_index(q, g)
    if gi is None:
        return None
    src = r["source"]
    if src == "hover" and g is False:                 # NOT_SUPPORTED = undetermined: supporting facts refute nothing
        return None
    if src == "wice_doc" and g == "partially_supported":
        return [SUP if k == gi else SKIP for k in range(len(opts))]
    return [SUP if k == gi else REF for k in range(len(opts))]


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default=str(Path(__file__).resolve().parent / "caps.json"), help="typed-stream caps")
    ap.add_argument("--drop_premises", default=str(ROOT / "data/v3/e15/snli_heldout_image_premises.json"),
                    help="E15 SNLI image-level split (premises kept out of training)")
    ap.add_argument("--drop_rows", default=str(ROOT / "data/v3/e15/minhash_drop_narrow.json"),
                    help="E15 MinHash near-duplicate drop list ([stream, row id])")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    (PROD / "targets").mkdir(parents=True, exist_ok=True)
    dprem = set(json.load(open(args.drop_premises)))
    drows = {tuple(x) for x in json.load(open(args.drop_rows))}

    def kept(state):   # = train_joint.kept (E15 SNLI image-level split)
        s = state.strip()
        return s not in dprem and s.split("\n")[0].removeprefix("Text 1: ").strip() not in dprem

    rows, L, excl = {}, {}, Counter()
    for pool in POOLS:
        L[pool] = load_lengths(pool)
        rs = []
        for r in jl(POOL_PATH[pool]):
            if not kept(r["state"]):
                excl[f"{pool}:snli_image_premise"] += 1
            elif (f"pool_{pool}", r["id"]) in drows:
                excl[f"{pool}:minhash_drop"] += 1
            else:
                rs.append(r)
        rows[pool] = rs
        log(f"pool {pool}: {len(rs)} rows kept")
    DTR.update({r["source"] for r in rows["train"]} - {"typed_decisions"})

    # ---------------------------------------------------------------- held-out long-state eval (a25)
    split_f = PROD / "eval_a25_split.json"
    if split_f.exists():   # frozen once made: re-runs (new teachers) must not move rows between train and eval
        split = json.load(open(split_f))["split"]
        split_stats = json.load(open(split_f))["stats"]
        log(f"eval split: reusing {split_f.name} ({len(split)} rows)")
    else:
        train_states = (r["state"] for p in ("train", "a23", "a23anli") for r in rows[p])
        split, split_stats = make_split(rows["a25"], L["a25"], train_states)
        json.dump({"split": split, "stats": split_stats}, open(split_f, "w"))
    for sp in ("dev", "test"):
        with open(PROD / f"eval_a25_{sp}.jsonl", "w") as f:
            for r in rows["a25"]:
                if split.get(r["id"]) == sp:
                    s, _, _, cut = L["a25"][r["id"]]
                    f.write(json.dumps({**r, "group": group_key(r), "state_tokens": s,
                                        "in_window": {qid: in_window(r, qid, s, cut) for qid in r["questions"]}},
                                       ensure_ascii=False) + "\n")
    rows["a25"] = [r for r in rows["a25"] if r["id"] not in split]
    log(f"eval split: dev {sum(v == 'dev' for v in split.values())} / test {sum(v == 'test' for v in split.values())} rows")

    # ---------------------------------------------------------------- generic questions (for mismatch packs)
    n_states = defaultdict(set)
    for pool in POOLS:
        for r in rows[pool]:
            hs = hash(r["state"])
            for qid, q in r["questions"].items():
                ins = instr_of(q, qid)
                if len(n_states[ins]) < GENERIC_STATES:
                    n_states[ins].add(hs)
    generic = {ins for ins, s in n_states.items() if len(s) >= GENERIC_STATES}
    del n_states
    log(f"generic instructions: {len(generic)}")

    # ---------------------------------------------------------------- items with calibrated-teacher inputs
    items = []
    cov = Counter()
    for pool in POOLS:
        ans = {t: load_teacher(t, pool) for t in TEACHERS}
        cov.update({f"{t}.{pool}": len(a) for t, a in ans.items()})
        for r in rows[pool]:
            s, _, _, cut = L[pool][r["id"]]
            fb = fit_bucket(r["id"])
            cl = cluster(r["source"])
            for qid, q in r["questions"].items():
                p = {}
                for t in TEACHERS:
                    a = ans[t].get(r["id"])
                    v = probs_of(q, a.get(qid)) if a else None
                    if v is not None and len(v) == len(options(q)):
                        p[t] = v
                gi = gold_index(q, r["gold"].get(qid))
                k = len(options(q))
                items.append({"pool": pool, "row": r["id"], "qid": qid, "type": q["type"], "src": r["source"],
                              "cl": cl, "fb": fb, "opts": options(q), "p": p, "dtr": r["source"] in DTR,
                              "gold": gi if gi is not None and 0 <= gi < k else None,
                              "win": in_window(r, qid, s, cut), "gq": instr_of(q, qid) in generic})
        del ans
        log(f"items after {pool}: {len(items)}")

    # calibration: per teacher x type on the fit bucket (E09 recipe, train split only)
    Ts = {}
    for t in TEACHERS:
        for ty in ("noul", "choice", "score"):
            sub = [it for it in items if it["fb"] == "fit" and it["type"] == ty and t in it["p"]
                   and it["gold"] is not None]
            Ts[f"{t}/{ty}"] = fit_T([np.log(np.clip(it["p"][t], 1e-4, 1)) for it in sub],
                                    [it["gold"] for it in sub]) if len(sub) >= 200 else 1.0
    log("temperatures", Ts)
    for it in items:
        for t in list(it["p"]):
            it["p"][t] = apply_T(it["p"][t], Ts[f"{t}/{it['type']}"])

    # stacking: 'all' on every fit item, then each cluster on its own items, shrunk toward its parent's fit
    own = defaultdict(list)
    for it in items:
        if it["fb"] == "fit" and it["gold"] is not None and it["p"]:
            own[it["cl"]].append(it)
            own["all"].append(it)
    levels = sorted({c for it in items for c in chain(it["cl"])}, key=lambda c: (0 if c == "all" else len(chain(c)), c))
    th, fitn = {arm: {} for arm in ARMS}, {}
    for c in levels:
        par = chain(c)[1] if c != "all" else None
        fitn[c] = len(own[c])
        d = pack(own[c]) if fitn[c] >= MIN_FIT else None
        for arm in ARMS:
            prior = th[arm][par] if par else torch.tensor([0.0, 0.0, 0.0, 0.0] if ARMS[arm][0] == "lin" else
                                                          [0.54, 0.54, 0.54, 0.0])
            th[arm][c] = fit_pool(d, arm, prior) if d is not None else prior.clone()
        log(f"stack {c}: n_fit {fitn[c]}")
    dev = [it for it in items if it["fb"] == "dev" and it["gold"] is not None and it["p"]]
    table = nll_table(dev, th)

    def params(c, arm):
        v = th[arm][c]
        w = torch.softmax(v[:-1], 0) if ARMS[arm][0] == "lin" else torch.nn.functional.softplus(v[:-1])
        return {"weights": dict(zip(TEACHERS, [round(float(x), 4) for x in w])), "T": round(float(torch.exp(v[-1])), 4)}
    stack_params = {c: {"n_fit": fitn[c], **{arm: params(c, arm) for arm in ARMS}} for c in levels}
    json.dump({"temperatures": Ts, "target_arm": TARGET_ARM, "params": stack_params, "dev_nll": table,
               "teacher_coverage": cov, "decider_trained_sources": sorted(DTR)},
              open(RES / "stack.json", "w"), indent=1)
    log("stack done; dev NLL all/all:", table["all"].get("all"))

    # ---------------------------------------------------------------- targets
    # expected committee probability of the gold option per cluster, for gold rows no teacher answered yet
    pg = {}
    for c in levels:
        if own[c]:
            d = pack(own[c])
            pg[c] = float(predict(th[TARGET_ARM][c], d, TARGET_ARM).gather(1, d[3][:, None]).exp().mean())
    by_cl = defaultdict(list)
    for i, it in enumerate(items):
        by_cl[it["cl"]].append(i)
    stacked = [None] * len(items)
    for cl, idx in by_cl.items():
        for j in range(0, len(idx), 50000):
            part = [items[i] for i in idx[j:j + 50000]]
            has = [n for n, it in enumerate(part) if it["p"]]
            if not has:
                continue
            P = predict(th[TARGET_ARM][cl], pack([part[n] for n in has]), TARGET_ARM).exp().numpy()
            for n, pr in zip(has, P):
                stacked[idx[j + n]] = pr[: len(part[n]["opts"])].astype(np.float64)
    del by_cl

    out_rows = defaultdict(dict)
    tstats = Counter()
    for it, ps in zip(items, stacked):
        k = len(it["opts"])
        oh = np.eye(k)[it["gold"]] if it["gold"] is not None else None
        ini = "".join(t[0] for t in TEACHERS if t in it["p"])            # j / d / c
        if ps is not None:
            ps = ps / ps.sum()
            tgt = 0.5 * oh + 0.5 * ps if oh is not None else ps
            src_code = ("g+" if oh is not None else "") + ini
        elif oh is not None:
            c = next((c for c in chain(it["cl"]) if c in pg), "all")
            cbar = pg[c]
            tgt = oh * (0.5 + 0.5 * cbar) + (1 - oh) * (0.5 * (1 - cbar) / max(1, k - 1))
            src_code = "g"
        else:
            tstats["no_target"] += 1
            continue
        tstats[f"ts_{src_code}"] += 1
        # abstention target: committee probability of missing the gold answer, else of missing its own argmax
        base = ps if ps is not None else tgt
        ab = 1 - float(base[it["gold"]]) if it["gold"] is not None else 1 - float(base.max())
        dis = None
        if ps is not None and len(it["p"]) > 1:
            am = int(ps.argmax())
            dis = round(float(np.mean([int(np.argmax(v)) != am for v in it["p"].values()])), 3)
        rec = {"t": [round(float(x), 4) for x in tgt], "ts": src_code, "ab": round(ab, 4)}
        if dis is not None:
            rec["dis"] = dis
        if it["type"] != "score":   # typed stance targets, undetermined split (D8, --stance_undet 1)
            d = float(tgt.max())
            rec["st"] = [[round(float(x), 4), round(float((1 - x) * d), 4), round(float((1 - x) * (1 - d)), 4)]
                         for x in tgt]
        if not it["win"]:
            rec["w"] = 0
        if it["gq"]:
            rec["gq"] = 1
        out_rows[(it["pool"], it["row"])][it["qid"]] = rec
    log("targets:", dict(tstats))
    del stacked

    # ---------------------------------------------------------------- evidence stream (a25)
    ev_rows, estats = [], Counter()
    for r in rows["a25"]:
        if not r.get("evidence"):
            continue
        s, _, _, cut = L["a25"][r["id"]]
        ws = None
        qs = []
        for qid, spans in r["evidence"].items():
            estats["questions_with_evidence"] += 1
            if s > WINDOW - 2 and max(e for _, e in spans) > cut:
                estats["outside_window"] += 1
                continue
            lab = evidence_labels(r, qid)
            if lab is None:
                estats["no_state_role"] += 1
                continue
            ws = ws or word_spans(r["state"])
            w = words_in(spans, ws)
            if not w:
                estats["empty_words"] += 1
                continue
            qs.append({"qid": qid, "lab": lab, "words": w})
            estats["kept"] += 1
            estats["blocks_support"] += sum(x == SUP for x in lab)
            estats["blocks_conflict"] += sum(x == REF for x in lab)
        if qs:
            ev_rows.append({"id": r["id"], "source": r["source"], "questions": qs})
    with gzip.open(PROD / "evidence_a25.jsonl.gz", "wt") as f:
        for x in ev_rows:
            f.write(json.dumps(x) + "\n")
    estats["rows"] = len(ev_rows)
    log("evidence:", dict(estats))

    # ---------------------------------------------------------------- selection (typed-stream caps)
    caps = json.load(open(args.caps))
    by_src = defaultdict(list)
    for pool in POOLS:
        for r in rows[pool]:
            q = out_rows.get((pool, r["id"]))
            if not q:
                continue
            good = {k: v for k, v in q.items() if v.get("w", 1)}
            if not good:
                continue
            nt = min(len(set(v["ts"].replace("g+", "").replace("g", ""))) for v in good.values())
            by_src[r["source"]].append((pool, r["id"], len(good), nt))
    keep, sel = set(), {}
    for src, lst in sorted(by_src.items()):
        cap = caps["per_source"].get(src, caps["default_cap"])
        lst.sort(key=lambda x: (-x[3], h01(x[1], "e34sel")))       # most teachers first, then a fixed random order
        n = 0
        for pool, rid, nq, _ in lst:
            if cap is not None and n >= cap:
                break
            keep.add((pool, rid))
            n += nq
        sel[src] = {"family": family(src), "cluster": cluster(src), "available_decisions": sum(x[2] for x in lst),
                    "kept_decisions": n, "kept_rows": sum(1 for x in lst if (x[0], x[1]) in keep), "cap": cap}
    files = {}
    for pool in POOLS:
        fn = PROD / "targets" / f"{pool}.jsonl.gz"
        n = 0
        with gzip.open(fn, "wt") as f:
            for r in rows[pool]:
                q = out_rows.get((pool, r["id"]))
                if not q:
                    continue
                rec = {"id": r["id"], "q": q}
                if (pool, r["id"]) in keep:
                    rec["k"] = caps.get("repeat", {}).get(r["source"], 1)
                if pool == "a25":
                    rec["g"] = group_key(r)
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
        files[pool] = {"rows": n, "bytes": fn.stat().st_size}
    manifest = {
        "built": time.strftime("%Y-%m-%d %H:%M"), "pools": {p: str(POOL_PATH[p].relative_to(ROOT)) for p in POOLS},
        "targets": files, "evidence": "data/v3/prod/evidence_a25.jsonl.gz", "teachers": TEACHERS,
        "teacher_coverage": cov, "temperatures": Ts, "target_arm": TARGET_ARM, "stack": stack_params, "window": WINDOW,
        "eval_split": {"file": "data/v3/prod/eval_a25_split.json", "stats": split_stats},
        "excluded": dict(excl), "target_sources": dict(tstats), "evidence_stats": dict(estats),
        "selection": sel, "caps": caps,
        "notes": "target = 0.5 gold + 0.5 stacked committee (gold rows), stacked committee (no gold), expected-mixc "
                 "prior (gold, no teacher yet: ts 'g'); st = typed stance [support, refute, undetermined] per option "
                 "(the --stance_undet 1 split of t); ab = abstention target; w = 0: not decidable within the window "
                 "(never trained); gq = 1: generic instruction (asked of >= 20 states; excluded from mismatch packs "
                 "under --mm_group 1); k = 1: selected for the typed stream; g = document group (a25)",
    }
    json.dump(manifest, open(PROD / "manifest.json", "w"), indent=1)
    log("manifest written;", {p: v for p, v in files.items()})


if __name__ == "__main__":
    main()
