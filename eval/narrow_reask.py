# SPDX-License-Identifier: MIT
"""E38a: narrow-and-reask ("effort 1") against a running server, no training.

Pass 1 asks the question on the full state with evidence on (both channels, every option, low threshold).
The narrowed state = the sentences holding any evidence span, plus one sentence of context each side, in document
order. Pass 2 re-asks the same question on that narrowed state. Text states only (JSON states are short already).

    python eval/narrow_reask.py --bench sets/bench.jsonl [--sets jevbench_original,jevbench_hard,aggrefact] \
        [--model horopter] [--out results/raw/narrow.jsonl]
-> resumable JSONL (one record per row: p1, p2, span counts, lengths)

The bench file is an eval set and is not shipped (see eval/e37_score.py).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import ask  # noqa: E402


def sentences(text):
    """[(start, end)] sentence char spans: split after . ! ? or at newlines."""
    out, s = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        if m.start() > s:
            out.append((s, m.start()))
        s = m.end()
    if s < len(text):
        out.append((s, len(text)))
    return out


def narrow(text, spans, ctx=1):
    sents = sentences(text)
    keep = set()
    for sp in spans:
        for i, (a, b) in enumerate(sents):
            if a < sp["end"] and sp["start"] < b:
                keep.update(range(max(0, i - ctx), min(len(sents), i + ctx + 1)))
    return " ".join(text[a:b] for i, (a, b) in enumerate(sents) if i in keep)


def compact(ans):
    a = ans["answers"]["q"]
    p = {"noul": a["noul"]} if "noul" in a else {"probabilities": a["probabilities"]}
    return {**p, "abstain": a["mbnli"]["abstain"]["p"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="E37 bench JSONL (eval only, not shipped)")
    ap.add_argument("--sets", default="jevbench_original,jevbench_hard,aggrefact")
    ap.add_argument("--model", default="horopter")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--out", default="results/raw/narrow.jsonl")
    a = ap.parse_args()
    sets = set(a.sets.split(","))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    rows = [r for r in map(json.loads, open(a.bench)) if r["set"] in sets and isinstance(r["state"], str)]
    with open(out, "a") as f:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            rec = {"id": r["id"], "set": r["set"]}
            try:
                o1, ms1 = ask(r["state"], r["questions"], a.model, a.threshold)
                spans = o1["answers"]["q"]["mbnli"]["evidence"]
                st2 = narrow(r["state"], spans)
                rec.update(p1=compact(o1), ms1=round(ms1, 1), n_spans=len(spans),
                           len_full=len(r["state"]), len_narrow=len(st2))
                if st2.strip():
                    o2, ms2 = ask(st2, r["questions"], a.model)
                    rec.update(p2=compact(o2), ms2=round(ms2, 1))
            except Exception as e:  # noqa: BLE001  keep going; the scorer counts missing pass 2 as "no change"
                rec["error"] = str(e)[:300]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    print("NARROW_REASK_DONE", flush=True)


if __name__ == "__main__":
    main()
