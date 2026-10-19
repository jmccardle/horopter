# SPDX-License-Identifier: MIT
"""E38d: training rows for a "check" adapter: the policy interpreter's single-check questions (one noul per check, plus the
exemption question), from policy_gen.py with a different seed than the policy_v0 eval set.
Domain robot_sop is held out of training, so the eval also measures a domain the adapter never saw.
Both registers: the prose state and the JSON state (rendered as JSON text, as the server does).

    python training/data/node_data.py --n 6000 --seed 1 --holdout robot_sop \
        > $HOROPTER_DATA/data/v3/e38/check_train.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_gen as G  # noqa: E402

SOFT = 0.97   # soft target: exact labels, mild smoothing as the E24 family files have


def rows_of(it, rng):
    qs, gold = {}, {}
    for i, n in enumerate(it["nodes"]):
        qs[f"n{i}"] = {"type": "noul", "instructions": n["instructions"]}
        gold[f"n{i}"] = n["gold"]
    if it["exempt"]:
        qs["ex"] = {"type": "noul", "instructions": f"Is this case one of the {it['exempt_who']}?"}
        gold["ex"] = bool(it["json_state"]["case"][it["exempt_field"]])
    g = {k: {"probabilities": {"true": SOFT if v else 1 - SOFT, "false": 1 - SOFT if v else SOFT}} for k, v in gold.items()}
    for reg, state in (("prose", it["prose_state"]), ("json", json.dumps(it["json_state"]))):
        yield {"id": f"{it['id']}/{reg}", "source": "e38_policy_check", "state": state, "questions": qs, "gold": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout", default="robot_sop")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = []
    i = 0
    while len(out) < 2 * a.n:
        for it in G.make(rng, 100000 + i):
            if it["domain"] != a.holdout:
                out += list(rows_of(it, rng))
        i += 1
    random.Random(a.seed).shuffle(out)
    for r in out:
        sys.stdout.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
