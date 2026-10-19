# SPDX-License-Identifier: MIT
"""E37: score every system's raw answers on the E37 bench (JevBench easy / original / hard, JF100, LLM-AggreFact,
E0 test). The bench is eval-only and is NOT shipped (LLM-AggreFact's terms bar training; JevBench / JF100 items stay
private): pass your own copy with --bench.

    python eval/e37_score.py --bench sets/bench.jsonl [--raw results/raw] [--out results] [--systems systems.json]
-> <out>/scores.json, <out>/scores.md

Raw answers: <raw>/<system>.jsonl, {"id", "answers": {qid: typed wire answer}} (one line per bench row; collect them
with eval/client.py against a server). Systems: --systems, a JSON list of [name, label, params, kind]; default = every
<raw>/*.jsonl file (label = name). Optional extra sources (the research runs of other systems): --e0_raw DIR with
<file>.choice4.test.jsonl E0 runs mapped by --e0_map JSON {system: file stem}; --e28_raw DIR with <system>.evalsets.jsonl
and <system>.choice4.test.jsonl for the systems in --e28 (comma list). --decontam: the 13-gram flagged-id file.
Metrics:
  jevbench_* / jf100   accuracy (argmax vs gold); JF100 also pair consistency (both items of a pair right)
  aggrefact            balanced accuracy at P(true) >= 0.5 per dataset, macro over datasets; also the macro over the
                       9 datasets outside the prod a25 evidence stream (RAGTruth and WiCE excluded)
  e0_test              4-way accuracy and macro-F1; IRRELEVANT-vs-rest AUROC from p(unrelated)
Missing answers count as wrong and are reported (coverage).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path("results/raw")     # --raw
E00_RAW = None                # --e0_raw
E00_NAME = {}                 # --e0_map
E28_RAW = None                # --e28_raw
E28 = set()                   # --e28
IN_A25 = {"RAGTruth", "Wice"}   # AggreFact datasets whose sources are in the a25 evidence stream
FLAGGED = set()               # --decontam
SYSTEMS = []                  # name, label, params, kind (--systems, else every raw file)


def configure(a):
    """Set the module paths from parsed args (also used by e37_family_breakdown.py)."""
    global RAW, E00_RAW, E00_NAME, E28_RAW, E28, FLAGGED, SYSTEMS
    RAW = Path(a.raw)
    E00_RAW = Path(a.e0_raw) if a.e0_raw else None
    E00_NAME = json.load(open(a.e0_map)) if a.e0_map else {}
    E28_RAW = Path(a.e28_raw) if a.e28_raw else None
    E28 = set(a.e28.split(",")) if a.e28 else set()
    FLAGGED = set(json.load(open(a.decontam))["flagged_ids"]) if a.decontam and Path(a.decontam).exists() else set()
    if a.systems:
        SYSTEMS = [tuple(x) for x in json.load(open(a.systems))]
    else:
        SYSTEMS = [(p.stem, p.stem, "", "") for p in sorted(RAW.glob("*.jsonl"))]
        SYSTEMS += [(n, n, "", "") for n in sorted(E28 | set(E00_NAME)) if n not in {s[0] for s in SYSTEMS}]


def add_args(ap):
    ap.add_argument("--bench", required=True, help="E37 bench JSONL (eval only, not shipped)")
    ap.add_argument("--raw", default="results/raw", help="dir of <system>.jsonl raw answers")
    ap.add_argument("--out", default="results")
    ap.add_argument("--systems", default=None, help="JSON list of [name, label, params, kind]")
    ap.add_argument("--decontam", default=None, help="JSON {flagged_ids: [...]} (13-gram overlap with training)")
    ap.add_argument("--e0_raw", default=None)
    ap.add_argument("--e0_map", default=None, help="JSON {system: E0 run file stem}")
    ap.add_argument("--e28_raw", default=None)
    ap.add_argument("--e28", default="", help="comma list of systems read from --e28_raw")


def probs(q, a):
    if q["type"] == "noul":
        p = float(a["noul"]) if "noul" in a else float(a.get("probabilities", {}).get("true", np.nan))
        return [True, False], np.array([p, 1 - p])
    names = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    return names, np.array([float(a["probabilities"][n]) for n in names])


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def load_raw(system):
    ans = {}
    p = RAW / f"{system}.jsonl"
    if p.exists():
        for l in open(p):
            r = json.loads(l)
            if "answers" in r:
                ans[r["id"]] = r["answers"]
    if system in E28 and E28_RAW:   # E28 llama.cpp runs: same JevBench / JF100 rows and E0 choice4 test
        for l in open(E28_RAW / f"{system}.evalsets.jsonl"):
            r = json.loads(l)
            if r.get("probs") is None:
                continue
            names = r["names"]
            names = [str(n) for n in names]
            a = ({"noul": r["probs"][names.index("True")]} if "True" in names else
                 {"probabilities": {str(n): p for n, p in zip(names, r["probs"])}})
            ans[f"{r['set']}/{r['id']}"] = {"q": a}
        for l in open(E28_RAW / f"{system}.choice4.test.jsonl"):
            r = json.loads(l)
            ans.setdefault(r["id"], {"q": {"type": "choice", "probabilities": r["probs"]}})
    if system in E00_NAME and E00_RAW and (E00_RAW / f"{E00_NAME[system]}.choice4.test.jsonl").exists():
        for l in open(E00_RAW / f"{E00_NAME[system]}.choice4.test.jsonl"):
            r = json.loads(l)
            ans.setdefault(r["id"], {"q": {"type": "choice", "probabilities": r["probs"]}})
    return ans


def score(rows, ans):
    by_set = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    out = {}
    for s, rs in by_set.items():
        recs = []
        for r in rs:
            q = r["questions"]["q"]
            a = (ans.get(r["id"]) or {}).get("q") or next(iter((ans.get(r["id"]) or {}).values()), None)
            g = r["gold"]["q"]
            try:
                names, p = probs(q, a)
                ok = bool(np.isfinite(p).all())
            except Exception:  # noqa: BLE001  missing / malformed answer
                names, p, ok = None, None, False
            recs.append((r, names, p, ok, g))
        cov = float(np.mean([x[3] for x in recs]))
        right = [bool(ok and str(names[int(np.argmax(p))]) == str(g)) for r, names, p, ok, g in recs]
        res = {"n": len(recs), "coverage": round(cov, 4)}
        if s == "aggrefact":
            per = {}
            for d in sorted({r["dataset"] for r in rs}):
                sub = [x for x in recs if x[0]["dataset"] == d]
                tp = [x[3] and x[2][0] >= 0.5 for x in sub if x[4]]
                tn = [x[3] and x[2][0] < 0.5 for x in sub if not x[4]]
                per[d] = round(100 * 0.5 * (np.mean(tp) + np.mean(tn)), 1)
            res["bacc_per_dataset"] = per
            res["bacc_macro"] = round(float(np.mean(list(per.values()))), 1)
            res["bacc_macro_outside_a25"] = round(float(np.mean([v for k, v in per.items() if k not in IN_A25])), 1)
            clean = {}
            for d in sorted({r["dataset"] for r in rs}):   # 13-gram clean items only (decontam.py)
                sub = [x for x in recs if x[0]["dataset"] == d and x[0]["id"] not in FLAGGED]
                tp = [x[3] and x[2][0] >= 0.5 for x in sub if x[4]]
                tn = [x[3] and x[2][0] < 0.5 for x in sub if not x[4]]
                if len(tp) >= 20 and len(tn) >= 20:
                    clean[d] = round(100 * 0.5 * (np.mean(tp) + np.mean(tn)), 1)
            res["bacc_macro_clean"] = round(float(np.mean(list(clean.values()))), 1)
            res["clean_datasets"] = sorted(clean)
        else:
            res["acc"] = round(100 * float(np.mean(right)), 1)
            cl = [ok for (r, *_), ok in zip(recs, right) if r["id"] not in FLAGGED]
            if len(cl) < len(right):
                res["acc_clean"] = round(100 * float(np.mean(cl)), 1)
                res["n_clean"] = len(cl)
        if s == "jf100":
            pair = defaultdict(list)
            for (r, *_), ok in zip(recs, right):
                pair[r["pair"]].append(ok)
            res["pair_consistency"] = round(100 * float(np.mean([all(v) for v in pair.values()])), 1)
        if s == "e0_test":
            labs = ["supports", "contradicts", "related_but_undetermined", "unrelated"]
            pred = [names[int(np.argmax(p))] if ok else None for r, names, p, ok, g in recs]
            gold = [x[4] for x in recs]
            f1 = []
            for c in labs:
                tp = sum(pp == c and gg == c for pp, gg in zip(pred, gold))
                f1.append(0.0 if tp == 0 else 2 * tp / (sum(pp == c for pp in pred) + sum(gg == c for gg in gold)))
            res["macro_f1"] = round(100 * float(np.mean(f1)), 1)
            ok_recs = [x for x in recs if x[3]]
            res["irrelevant_auroc"] = round(auroc([x[4] == "unrelated" for x in ok_recs],
                                                  [x[2][x[1].index("unrelated")] for x in ok_recs]), 3)
        out[s] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    a = ap.parse_args()
    configure(a)
    out = Path(a.out)
    rows = [json.loads(l) for l in open(a.bench)]
    res = {}
    for name, label, params, kind in SYSTEMS:
        ans = load_raw(name)
        if ans:
            res[name] = {"label": label, "params": params, "kind": kind, "sets": score(rows, ans)}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "scores.json", "w"), indent=1)
    cols = [("jevbench_easy", "acc"), ("jevbench_original", "acc"), ("jevbench_hard", "acc"), ("jf100", "acc"),
            ("jf100", "pair_consistency"), ("aggrefact", "bacc_macro"), ("aggrefact", "bacc_macro_clean"),
            ("e0_test", "acc"), ("e0_test", "acc_clean"), ("e0_test", "macro_f1"), ("e0_test", "irrelevant_auroc")]
    md = ["| system | params | " + " | ".join(f"{s} {m}" for s, m in cols) + " | coverage (min) |",
          "|---|---|" + "---|" * (len(cols) + 1)]
    for name, r in res.items():
        cells = [str(r["sets"][s].get(m, "—")) if r["sets"].get(s, {}).get("coverage") else "—" for s, m in cols]
        cov = min(v["coverage"] for v in r["sets"].values() if v["coverage"])
        md.append(f"| {r['label']} | {r['params']} | " + " | ".join(cells) + f" | {cov} |")
    agg = [name for name in res if "aggrefact" in res[name]["sets"]]
    if agg:
        ds = list(res[agg[0]]["sets"]["aggrefact"]["bacc_per_dataset"])
        md += ["", "AggreFact balanced accuracy per dataset (RAGTruth and WiCE are in the prod a25 evidence stream):", "",
               "| system | " + " | ".join(ds) + " |", "|---|" + "---|" * len(ds)]
        for name in agg:
            per = res[name]["sets"]["aggrefact"]["bacc_per_dataset"]
            if res[name]["sets"]["aggrefact"]["coverage"]:
                md.append(f"| {res[name]['label']} | " + " | ".join(str(per[d]) for d in ds) + " |")
    (out / "scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
