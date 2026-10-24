# SPDX-License-Identifier: MIT
"""Router training data for your own adapters: one routing question per sampled training question, asking which of
your targets should answer it (the question layout: the routed question's own state, the question embedded in the
router question, as the server asks it; horopter.routing).

    python training/build_router_data.py --tokenizer export/horopter-1B --out data/router \\
        --target refunds "a retail refund or return decision" data/refunds_train.jsonl \\
        --target triage "which support team gets a ticket" data/triage_train.jsonl [--per_target 4000]
    python training/train_adapter.py --trunk export/horopter-1B --name router --data data/router/router_train.jsonl \\
        --meta data/router/router_meta.json --steps 1500

Each --target: a name (the adapter's name when served), a concrete one-line description of its questions (E51: the
router reads these; "the sentiment of a financial news headline" routes far better than "labelling a text"), and the
training rows of that adapter ({id, state, questions}: the train_adapter.py format). Options are shuffled per row, and
2/3 of rows keep a random subset of 2 or more targets (always the right one), so the router learns to pick among the
targets it is given rather than fixed slots (E51: 100% with options shuffled or removed).
-> <out>/router_train.jsonl (every row; train_adapter.py holds out its last --dev_n rows), router_dev.jsonl (those
   last --dev_n rows, for your own checks), router_meta.json ({"routing_layout": "question", "targets": {name: description}}) for adapter.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horopter import routing as RT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", nargs=3, action="append", required=True, metavar=("NAME", "DESCRIPTION", "ROWS"))
    ap.add_argument("--tokenizer", required=True, help="the model directory or hub id (its tokenizer cuts questions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_target", type=int, default=4000, help="routing questions per target")
    ap.add_argument("--max_q", type=int, default=128, help="the model's question block (config.json max_q)")
    ap.add_argument("--dev_n", type=int, default=300, help="= train_adapter.py --dev_n")
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    if len(a.target) < 2:
        ap.error("a router needs at least 2 targets")
    targets = {n: d for n, d, _ in a.target}
    if len(targets) < len(a.target):
        ap.error("target names must differ")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rng = random.Random(a.seed)
    names = list(targets)
    rows, ncut = [], 0
    for name, _, path in a.target:
        pairs = [(r, qid) for r in map(json.loads, open(path)) for qid in r["questions"]]
        for r, qid in rng.sample(pairs, min(a.per_target, len(pairs))):
            keys = names[:]
            if len(names) > 2 and rng.random() < 2 / 3:
                keys = [name] + rng.sample([k for k in names if k != name], rng.randint(1, len(names) - 1))
            rng.shuffle(keys)
            rt = RT.Routing(router="router", targets={k: targets[k] for k in keys}, layout="question")
            q, cut = rt.question(r["questions"][qid], tok, a.max_q)
            ncut += cut
            rows.append({"id": f"route:{name}:{r['id']}:{qid}", "source": f"route_{name}", "state": r["state"],
                         "questions": {"route": q}, "gold": {"route": {"probabilities": {k: float(k == name)
                                                                                        for k in keys}}}})
    rng.shuffle(rows)
    n_dev = min(a.dev_n, len(rows) // 5)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fname, part in (("router_train.jsonl", rows), ("router_dev.jsonl", rows[len(rows) - n_dev:])):
        with open(out / fname, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    json.dump({"routing_layout": "question", "targets": targets}, open(out / "router_meta.json", "w"), indent=1)
    print(f"{len(rows)} routing questions over {len(names)} targets ({n_dev} dev = the last rows of router_train; "
          f"{ncut} embedded questions cut to fit) -> {out}")


if __name__ == "__main__":
    main()
