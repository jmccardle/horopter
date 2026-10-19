# SPDX-License-Identifier: MIT
"""E37 addendum: accuracy per item family on JevBench hard and JF100 (the beta's gap sets), per system.
The baseline for the beta-2 sprint: which question shapes the gap is in.

    python eval/e37_family_breakdown.py --bench sets/bench.jsonl [--raw results/raw] [--out results] [--systems ...]
-> <out>/family.{json,md}   (same arguments as e37_score.py)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e37_score as S  # noqa: E402

SETS = ("jevbench_hard", "jf100")


def main():
    ap = argparse.ArgumentParser()
    S.add_args(ap)
    a = ap.parse_args()
    S.configure(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.bench)]
    rows = [r for r in rows if r["set"] in SETS]
    fams = {s: sorted({r["family"] for r in rows if r["set"] == s}) for s in SETS}
    res = {}
    for name, *_ in S.SYSTEMS:
        ans = S.load_raw(name)
        acc = defaultdict(list)
        for r in rows:
            a_ = (ans.get(r["id"]) or {}).get("q") or next(iter((ans.get(r["id"]) or {}).values()), None)
            try:
                names, p = S.probs(r["questions"]["q"], a_)
                ok = bool(np.isfinite(p).all()) and str(names[int(np.argmax(p))]) == str(r["gold"]["q"])
            except Exception:  # noqa: BLE001  missing answer counts as wrong, as in e37_score.py
                ok = False
            acc[(r["set"], r["family"])].append(ok)
        res[name] = {f"{s}/{f}": round(100 * float(np.mean(v)), 1) for (s, f), v in acc.items()}
    json.dump(res, open(out / "family.json", "w"), indent=1)
    names = [n for n, *_ in S.SYSTEMS]
    lines = ["# E37 accuracy per family (JevBench hard, JF100)", "",
             "| set / family | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for s in SETS:
        for f in fams[s]:
            n = sum(r["set"] == s and r["family"] == f for r in rows)
            lines.append(f"| {s} / {f} | {n} | " + " | ".join(str(res[m].get(f"{s}/{f}", "—")) for m in names) + " |")
    open(out / "family.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
