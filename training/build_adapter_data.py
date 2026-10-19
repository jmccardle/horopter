# SPDX-License-Identifier: MIT
"""Training data for the family adapters and the router (rows {id, source, state, questions, gold}).

  <out>/expert_<family>.jsonl     up to --cap rows of the family's training file (seeded sample)
  <out>/router_{train,dev}.jsonl  one router question per sampled (row, question): state = router_state(...),
                                  gold = one-hot family; --per_family questions per family, 10% dev
  <out>/source_family.json        training source -> family
  <out>/data_stats.json

The family training files are <families_dir>/<stem>.jsonl, stems in horopter.families.FAMILIES (typed_mix, nli_mix,
rel_mix, cls_mix, kqa_mix, clause_gold). Router examples are drawn from the same files (no eval rows).

    python training/build_adapter_data.py --families_dir $HOROPTER_DATA/data/v3/e20/train \
        --out $HOROPTER_DATA/data/v3/e24 [--cap 20000] [--per_family 4000]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horopter import families as F  # noqa: E402


def source_family(families_dir, out):
    """Training source -> family (the first family whose file carries the source)."""
    m = {}
    for fam, (fn, _) in F.FAMILIES.items():
        for line in open(Path(families_dir) / f"{fn}.jsonl"):
            m.setdefault(json.loads(line)["source"], fam)
    json.dump(m, open(Path(out) / "source_family.json", "w"), indent=1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--per_family", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    router, stats = [], {}
    for fam, (fn, _) in F.FAMILIES.items():
        rows = [json.loads(l) for l in open(Path(a.families_dir) / f"{fn}.jsonl")]
        rng.shuffle(rows)
        exp = rows[: a.cap]
        with open(out / f"expert_{fam}.jsonl", "w") as f:
            for r in exp:
                f.write(json.dumps(r) + "\n")
        pairs = [(r, qid) for r in rows for qid in r["questions"]]
        rng.shuffle(pairs)
        for r, qid in pairs[: a.per_family]:
            router.append({"id": f"route:{fam}:{r['id']}:{qid}", "source": f"route_{fam}",
                           "state": F.router_state(r["state"], r["questions"][qid]),
                           "questions": {"route": F.ROUTER_Q},
                           "gold": {"route": {"probabilities": {k: float(k == fam) for k in F.FAMILIES}}}})
        stats[fam] = {"expert_rows": len(exp), "expert_questions": sum(len(r["questions"]) for r in exp),
                      "router_questions": min(a.per_family, len(pairs))}
    rng.shuffle(router)
    n_dev = len(router) // 10
    for name, part in (("router_dev", router[:n_dev]), ("router_train", router[n_dev:])):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    stats["router"] = {"train": len(router) - n_dev, "dev": n_dev}
    source_family(a.families_dir, out)
    json.dump(stats, open(out / "data_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
