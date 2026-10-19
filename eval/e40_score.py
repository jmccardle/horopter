# SPDX-License-Identifier: MIT
"""E40: score raw answers (e40_run_eval.py) -> <out>/scores.{json,md}.

    python eval/e40_score.py [--data <sets dir>] [--raw results/raw] [--out results] \
        [--models horopter,horopter-base,horopter-check] [--short beta,base,check] [--routed horopter]

--models: the first three columns, in order: routed model, trunk only, check adapter (the research runs used
mbnli-3-beta, mbnli-3-beta-base, mbnli-3-beta-check). A decoder reference (qwen3-30b-a3b raw files, if present for
every set) is appended, as in the research scorer.

noul: predict "true" when P(true) >= 0.5. choice: argmax of probabilities.
Majority baseline = always answer the most frequent gold label of that slice.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(os.environ.get("HOROPTER_DATA", ".")) / "data/v3/e40"
RAW = Path("results/raw")
OUT = Path("results")
SETS = ["proofwriter_cwa_d5_dev", "proofwriter_owa_d5_dev", "sharc_dev", "conditionalqa_dev"]
MODELS = ["horopter", "horopter-base", "horopter-check"]
SHORT = {"horopter": "beta", "horopter-base": "base", "horopter-check": "check", "qwen3-30b-a3b": "qwen3-30b"}
ROUTED = "horopter"   # the routed model: its adapters_used are tallied


def configure():
    global DATA, RAW, OUT, MODELS, SHORT, ROUTED
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--short", default="beta,base,check", help="column names for --models")
    ap.add_argument("--routed", default=None, help="routed model (default: the first of --models)")
    a = ap.parse_args()
    DATA, RAW, OUT = Path(a.data), Path(a.raw), Path(a.out)
    MODELS = a.models.split(",")
    SHORT = dict(zip(MODELS, a.short.split(","))) | {"qwen3-30b-a3b": "qwen3-30b"}
    for m in MODELS:
        SHORT.setdefault(m, m)
    ROUTED = a.routed or MODELS[0]
    if all((RAW / f"{s}__qwen3-30b-a3b.jsonl").exists() for s in SETS):   # decoder reference
        MODELS = MODELS + ["qwen3-30b-a3b"]


def pred(a, keys=None):
    if "noul" in a:
        return "true" if a["noul"] >= 0.5 else "false"
    p = a["probabilities"]
    return max(keys or p, key=lambda k: p[k])


def acc(items):
    """items: [(pred, gold)] -> (accuracy %, n)"""
    return (round(100 * sum(p == g for p, g in items) / len(items), 1) if items else None, len(items))


def majority(golds):
    if not golds:
        return None
    return round(100 * Counter(golds).most_common(1)[0][1] / len(golds), 1)


def load(s, m):
    p = RAW / f"{s}__{m}.jsonl"
    out = {}
    if p.exists():
        for l in open(p):
            r = json.loads(l)
            out[r["id"]] = r
    return out


def items_of(s):
    """Flatten rows to question items: (row, qid, gold, slice-keys)."""
    return [json.loads(l) for l in open(DATA / f"{s}.jsonl")]


def main():
    configure()
    S, md = {}, ["# E40 scores", "", "Accuracy in %, n in brackets. noul: P(true) >= 0.5; choice: argmax.", ""]
    errors, adapters = Counter(), defaultdict(Counter)
    raw = {(s, m): load(s, m) for s in SETS for m in MODELS}
    for (s, m), d in raw.items():
        for r in d.values():
            if "error" in r:
                errors[(s, m)] += 1
            elif m == ROUTED:
                adapters[s][",".join(r.get("adapters_used") or ["none"])] += 1

    def collect(s, m, qsel, keys=None):
        """-> list of (pred, gold, row, qid) over answered questions chosen by qsel(row) -> [qid]."""
        out = []
        for r in rows[s]:
            a = raw[(s, m)].get(r["id"])
            if not a or "answers" not in a:
                continue
            for q in qsel(r):
                out.append((pred(a["answers"][q], keys), r["gold"][q], r, q))
        return out

    rows = {s: items_of(s) for s in SETS}
    allq = lambda r: list(r["questions"])

    # ---- ProofWriter depth curve
    for s in ["proofwriter_cwa_d5_dev", "proofwriter_owa_d5_dev"]:
        owa = "owa" in s
        res = {"by_depth": {}, "majority_by_depth": {}}
        golds = defaultdict(list)
        for r in rows[s]:
            for q in r["questions"]:
                golds[r["meta"]["depth"][q]].append(r["gold"][q])
        hdr = "| model | " + " | ".join(f"d{d}" for d in range(6)) + " | all |"
        md += [f"## {s}: accuracy by proof depth" + (" (3-way true/false/unknown)" if owa else " (true/false)"), "",
               hdr, "|" + "---|" * 8]
        maj = [majority(golds[d]) for d in range(6)]
        md.append("| majority baseline | " + " | ".join(str(x) for x in maj) + f" | {majority(sum(golds.values(), []))} |")
        res["majority_by_depth"] = dict(zip(range(6), maj))
        for m in MODELS:
            it = collect(s, m, allq)
            byd = {d: acc([(p, g) for p, g, r, q in it if r["meta"]["depth"][q] == d]) for d in range(6)}
            tot = acc([(p, g) for p, g, *_ in it])
            res["by_depth"][m] = {**{d: v[0] for d, v in byd.items()}, "all": tot[0], "n": tot[1]}
            md.append(f"| {SHORT[m]} | " + " | ".join(str(byd[d][0]) for d in range(6)) + f" | {tot[0]} ({tot[1]}) |")
            res.setdefault("pred_dist", {})[m] = dict(Counter(p for p, *_ in it))
        if owa:   # true/false-gold items only, argmax over the three options; and over true/false only
            md += ["", "OWA, items with gold true/false only (unknown items removed):", "",
                   "| model | readout | " + " | ".join(f"d{d}" for d in range(6)) + " | all |", "|" + "---|" * 9]
            tf = lambda r: [q for q in r["questions"] if r["gold"][q] != "unknown"]
            res["tf_only"] = {}
            for m in MODELS:
                for lab, keys in [("3-way argmax", None), ("true vs false", ["true", "false"])]:
                    it = collect(s, m, tf, keys)
                    byd = [acc([(p, g) for p, g, r, q in it if r["meta"]["depth"][q] == d])[0] for d in range(6)]
                    tot = acc([(p, g) for p, g, *_ in it])
                    res["tf_only"][f"{m} {lab}"] = byd + [tot[0]]
                    md.append(f"| {SHORT[m]} | {lab} | " + " | ".join(map(str, byd)) + f" | {tot[0]} ({tot[1]}) |")
            md += ["", "OWA prediction counts (all items): " + "; ".join(
                f"{SHORT[m]} {res['pred_dist'][m]}" for m in MODELS)]
        # gold x polarity ("not" in the statement): exposes the closed-world convention (CWA: absent => false)
        cells = [(g, pol) for g in (["true", "false", "unknown"] if owa else ["true", "false"]) for pol in ("pos", "neg")]
        md += ["", "By gold label x polarity (neg = statement contains 'not'), all depths:", "",
               "| model | " + " | ".join(f"{g}/{p}" for g, p in cells) + " |", "|" + "---|" * (len(cells) + 1)]
        res["gold_polarity"] = {}
        for m in MODELS:
            it = collect(s, m, allq)
            pol = lambda r, q: "neg" if " not " in r["questions"][q]["instructions"] else "pos"
            v = [acc([(p, g) for p, g, r, q in it if g == cg and pol(r, q) == cp]) for cg, cp in cells]
            res["gold_polarity"][m] = {f"{g}/{p}": x[0] for (g, p), x in zip(cells, v)}
            md.append(f"| {SHORT[m]} | " + " | ".join(f"{x[0]} ({x[1]})" for x in v) + " |")
        md.append("")
        S[s] = res

    # ---- ShARC
    s = "sharc_dev"
    res = {}
    g4 = [r["gold"]["q"] for r in rows[s]]
    gb = [r["gold"]["qb"] for r in rows[s] if "qb" in r["gold"]]
    md += ["## sharc_dev", "", "4-way = choice yes/no/irrelevant/more_info on every item. Binary subset = items with "
           "gold yes/no only: noul on the user question, and the 4-way choice read as yes vs no only.", "",
           "| model | 4-way | binary noul | binary from choice (yes vs no) | " +
           " | ".join(f"recall {k}" for k in ["yes", "no", "irrelevant", "more_info"]) + " |", "|" + "---|" * 8,
           f"| majority baseline | {majority(g4)} | {majority(gb)} | {majority(gb)} | | | | |"]
    for m in MODELS:
        it = collect(s, m, lambda r: ["q"])
        a4 = acc([(p, g) for p, g, *_ in it])
        ab = acc([(p, g) for p, g, *_ in collect(s, m, lambda r: ["qb"] if "qb" in r["gold"] else [])])
        ac = acc([(p, g) for p, g, *_ in collect(s, m, lambda r: ["q"] if r["meta"]["binary"] else [], ["yes", "no"])])
        rec = {k: acc([(p, g) for p, g, *_ in it if g == k])[0] for k in ["yes", "no", "irrelevant", "more_info"]}
        res[m] = {"4way": a4[0], "n": a4[1], "binary_noul": ab[0], "binary_n": ab[1], "binary_choice": ac[0],
                  "recall": rec, "pred_dist": dict(Counter(p for p, *_ in it))}
        md.append(f"| {SHORT[m]} | {a4[0]} ({a4[1]}) | {ab[0]} ({ab[1]}) | {ac[0]} | " +
                  " | ".join(str(rec[k]) for k in rec) + " |")
    md += ["", "ShARC by follow-up history depth (turns already in the state):", "",
           "| model | readout | " + " | ".join(f"h{d}" for d in range(5)) + " |", "|" + "---|" * 7]
    deps = range(5)
    md.append("| majority | 4-way | " + " | ".join(str(majority([r["gold"]["q"] for r in rows[s] if r["meta"]["depth"] == d])) for d in deps) + " |")
    md.append("| majority | binary | " + " | ".join(str(majority([r["gold"]["qb"] for r in rows[s] if r["meta"]["depth"] == d and "qb" in r["gold"]])) for d in deps) + " |")
    md.append("| n | 4-way / binary | " + " | ".join(
        f"{sum(r['meta']['depth'] == d for r in rows[s])} / {sum(r['meta']['depth'] == d and r['meta']['binary'] for r in rows[s])}" for d in deps) + " |")
    for m in MODELS:
        it4 = collect(s, m, lambda r: ["q"])
        itb = collect(s, m, lambda r: ["qb"] if "qb" in r["gold"] else [])
        b4 = [acc([(p, g) for p, g, r, q in it4 if r["meta"]["depth"] == d])[0] for d in deps]
        bb = [acc([(p, g) for p, g, r, q in itb if r["meta"]["depth"] == d])[0] for d in deps]
        res[m]["by_depth_4way"], res[m]["by_depth_binary"] = b4, bb
        md.append(f"| {SHORT[m]} | 4-way | " + " | ".join(map(str, b4)) + " |")
        md.append(f"| {SHORT[m]} | binary noul | " + " | ".join(map(str, bb)) + " |")
    md.append("")
    S[s] = res

    # ---- ConditionalQA
    s = "conditionalqa_dev"
    res = {}
    gall = [r["gold"]["q"] for r in rows[s]]
    gc = lambda c: [r["gold"]["q"] for r in rows[s] if r["meta"]["has_conditions"] == c]
    md += ["## conditionalqa_dev (yes/no questions, noul)", "",
           "| model | all | no conditions | with conditions | predicted-true rate |", "|---|---|---|---|---|",
           f"| majority baseline | {majority(gall)} | {majority(gc(False))} | {majority(gc(True))} | |"]
    for m in MODELS:
        it = collect(s, m, allq)
        a = acc([(p, g) for p, g, *_ in it])
        a0 = acc([(p, g) for p, g, r, q in it if not r["meta"]["has_conditions"]])
        a1 = acc([(p, g) for p, g, r, q in it if r["meta"]["has_conditions"]])
        pt = round(100 * sum(p == "true" for p, *_ in it) / max(1, len(it)), 1)
        res[m] = {"all": a[0], "n": a[1], "no_conditions": a0[0], "n0": a0[1], "with_conditions": a1[0], "n1": a1[1],
                  "pred_true_rate": pt}
        md.append(f"| {SHORT[m]} | {a[0]} ({a[1]}) | {a0[0]} ({a0[1]}) | {a1[0]} ({a1[1]}) | {pt} |")
    md.append("")
    S[s] = res

    # ---- summary
    md += ["## Summary: accuracy per set", "", "| set | majority | " + " | ".join(SHORT[m] for m in MODELS) + " |",
           "|---|---|---|---|---|"]
    summ = {}
    for s in SETS:
        if s.startswith("proofwriter"):
            vals = [S[s]["by_depth"][m]["all"] for m in MODELS]
            mj = majority([r["gold"][q] for r in rows[s] for q in r["questions"]])
        elif s == "sharc_dev":
            vals, mj = [S[s][m]["4way"] for m in MODELS], majority(g4)
        else:
            vals, mj = [S[s][m]["all"] for m in MODELS], majority(gall)
        summ[s] = {"majority": mj, **dict(zip(MODELS, vals))}
        md.append(f"| {s} | {mj} | " + " | ".join(map(str, vals)) + " |")
    md += ["", f"Router ({ROUTED}) adapters used: " + "; ".join(f"{s} {dict(c)}" for s, c in adapters.items()),
           "", "Request errors: " + (", ".join(f"{s}/{m} {n}" for (s, m), n in errors.items()) or "none")]
    S["summary"], S["adapters_used_by_router"], S["errors"] = summ, {s: dict(c) for s, c in adapters.items()}, \
        {f"{s}/{m}": n for (s, m), n in errors.items()}
    S["answered_rows"] = {f"{s}/{m}": len(d) for (s, m), d in raw.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(S, open(OUT / "scores.json", "w"), indent=1)
    open(OUT / "scores.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
