# SPDX-License-Identifier: MIT
"""E34 step 0: token lengths of every pool row under the Ettin tokenizer (CPU).

Per row: state tokens (untruncated, no specials), question-block tokens as FastCollator packs them (question block
plus one block per option, each min(len, MAX_Q - 1) + 1), block count, and for every state the char offset where a
W-token state window ends (W = 2048 and 512), so evidence spans can be checked against the window.

    HOROPTER_DATA=<data root> python training/data/lengths.py [pool ...]
-> $HOROPTER_DATA/data/v3/prod/lengths/<pool>.json.gz
"""
import gzip
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAX_Q, POOL_PATH, POOLS, PROD, instr_of, options  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

BASE = os.environ.get("HOROPTER_TOKENIZER", "jhu-clsp/ettin-encoder-1b")   # the Ettin sizes share one tokenizer
CUTS = (2048, 512)


def option_texts(q):   # = horopter.packing.option_texts (copied: importing it pulls in torch + the model code)
    NOUL_DEFAULT = {"true": "The statement holds.", "false": "The statement does not hold."}
    names = options(q)
    if q["type"] == "noul":
        cr = q.get("criteria") or NOUL_DEFAULT
        return [f"{n}: {cr.get(n, NOUL_DEFAULT[n])}" for n in names]
    if q["type"] == "score":
        return list(q["criteria"])
    return [n.replace("_", " ") if (d or n) == n else f"{n.replace('_', ' ')}: {d}" for n, d in q["criteria"].items()]


def block_texts(r):
    out = []
    for qid, q in r["questions"].items():
        ins = instr_of(q, qid)
        out.append(ins)
        out += [f"{ins} Answer: {o}" for o in option_texts(q)]
    return out


def main(pools):
    tok = AutoTokenizer.from_pretrained(BASE)
    (PROD / "lengths").mkdir(parents=True, exist_ok=True)
    qcache = {}
    for pool in pools:
        t0 = time.time()
        res = {"ids": [], "s": [], "q": [], "b": [], **{f"cut{w}": [] for w in CUTS}}
        buf = []

        def flush():
            states = [r["state"] for r in buf]
            enc = tok(states, add_special_tokens=False, return_offsets_mapping=True)
            new = list({t for r in buf for t in block_texts(r) if t not in qcache})
            if new:
                for t, ids in zip(new, tok(new, add_special_tokens=False)["input_ids"]):
                    qcache[t] = min(len(ids), MAX_Q - 1) + 1
            for r, ids, offs in zip(buf, enc["input_ids"], enc["offset_mapping"]):
                bt = block_texts(r)
                res["ids"].append(r["id"])
                res["s"].append(len(ids))
                res["q"].append(sum(qcache[t] for t in bt))
                res["b"].append(len(bt))
                for w in CUTS:
                    res[f"cut{w}"].append(offs[w - 3][1] if len(ids) > w - 2 else len(r["state"]))
            buf.clear()
        for line in open(POOL_PATH[pool]):
            buf.append(json.loads(line))
            if len(buf) >= 4000:
                flush()
        if buf:
            flush()
        with gzip.open(PROD / "lengths" / f"{pool}.json.gz", "wt") as f:
            json.dump(res, f)
        n = len(res["s"])
        print(f"[lengths] {pool}: {n} rows, state tokens mean {sum(res['s']) / n:.0f}, "
              f">2046 {sum(s > 2046 for s in res['s'])}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or POOLS)
