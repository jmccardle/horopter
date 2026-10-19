# SPDX-License-Identifier: MIT
"""E39: beta-2 retrain data (ROADMAP W2; user OK 2026-10-18: retrain after check v2, 400M first).

data/v3/prod2 = the E34 production data (data/v3/prod, built by build.py; unchanged pools / targets / lengths, linked)
plus one new pool
"e39" with three exact-label streams:
  policy   W2.1 generator (policy_gen.py, seed 3; robot_sop held out like every E38 adapter), both registers. Per row:
           the 4-option policy question and every single-check question (the interpreter's nodes), so the trunk
           itself learns the checks. Evidence (prose rows): the deciding rule sentence supports the gold option.
  check2   E38f Qwen-written policies with compiler-style conditions and hard negatives, blind-verified (the train
           part of check_train_v2.jsonl; the 10% dev policies stay out).
  cf       E38c counterfactual minimal pairs (v1 + v2, judge-verified): the original text with the gold answer and
           the edited text with the target answer. Evidence: the changed words support that side's answer.
Targets are exact labels smoothed to 0.97 (stance [support, refute, undetermined] = [t, 1 - t - 0.01, 0.01];
abstention = 1 - max t). Every new state is checked against the E37 bench (13-gram) and E38b's policy_v0 eval states.

Inputs (under --data_root, default $HOROPTER_DATA; none of them ship with the repo):
  data/v3/prod/                     the E34 build (manifest, targets, lengths, evidence_a25, eval_a25_*)
  data/v3/e38/check_train_v2.jsonl  check_v2_data.py assemble (optional: the check2 stream is empty without it)
  data/v3/e38/cf2_pairs.jsonl, cf_rows.jsonl, cf_edits.jsonl, cf_scored.jsonl   cf_pairs.py (v2 select2, v1 score)
  --policy_eval (default data/v3/e38/policy_v0.jsonl = policy_gen.py --seed 0): eval states kept out
  --bench       the E37 bench JSONL (eval only): 13-gram decontamination, required
Outputs: data/v3/prod2/ (manifest, targets/e39, lengths/e39, evidence_prod2, links to the prod files), data/v3/e39/pool_e39.jsonl

    python training/data/build_prod2.py --bench bench.jsonl [--n_policy 6000] [--k_cf 4]
"""
import argparse
import gzip
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(os.environ.get("HOROPTER_DATA", ".")).resolve()     # --data_root
SRC, OUT, D38 = ROOT / "data/v3/prod", ROOT / "data/v3/prod2", ROOT / "data/v3/e38"
POLICY_EVAL = None   # --policy_eval
SOFT = 0.97


def options(q):
    return ["true", "false"] if q["type"] == "noul" else list(q["criteria"])


def target(q, gold):
    """Smoothed one-hot over options(q) + stance rows + abstention."""
    opts = options(q)
    g = ("true" if gold in (True, "true") else "false") if q["type"] == "noul" else str(gold)
    lo = (1 - SOFT) / (len(opts) - 1)
    t = [SOFT if o == g else lo for o in opts]
    st = [[round(x, 4), round(1 - x - 0.01, 4), 0.01] for x in t]
    return {"t": [round(x, 4) for x in t], "ts": "g", "ab": round(1 - SOFT, 4), "dis": 0.0, "st": st}, opts.index(g)


def word_ids(text, spans):
    """Whitespace-word indices (text.split()) overlapping any char span."""
    out, pos = [], 0
    for i, w in enumerate(text.split()):
        pos = text.index(w, pos)
        if any(pos < e and pos + len(w) > s for s, e in spans):
            out.append(i)
        pos += len(w)
    return out


def ngrams(t, n=13):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def policy_rows(n, rng, k):
    import policy_gen as G
    eval_states = {json.loads(l)["prose_state"] for l in open(POLICY_EVAL)}
    rows, ev, i = [], [], 0
    while len(rows) < 2 * n:
        for it in G.make(rng, 300000 + i):
            if it["domain"] == "robot_sop" or it["prose_state"] in eval_states:
                continue
            q = it["questions"]["q"]
            qs, tg = {"q": q}, {}
            tg["q"], gi = target(q, it["gold"]["q"])
            for j, nd in enumerate(it["nodes"]):
                qn = {"type": "noul", "instructions": nd["instructions"]}
                qs[f"n{j}"] = qn
                tg[f"n{j}"] = target(qn, nd["gold"])[0]
            if it["exempt"]:
                qe = {"type": "noul", "instructions": f"Is this case one of the {it['exempt_who']}?"}
                qs["ex"] = qe
                tg["ex"] = target(qe, bool(it["json_state"]["case"][it["exempt_field"]]))[0]
            for reg, st in (("prose", it["prose_state"]), ("json", json.dumps(it["json_state"]))):
                rid = f"e39_policy/{it['id']}/{reg}"
                rows.append(({"id": rid, "source": "e39_policy", "permissions": "libre", "state": st, "questions": qs},
                             {"id": rid, "q": tg, "k": k, "g": f"e39_policy|{it['pair']}"}))
                if reg == "prose" and it["gold_span"]:
                    lab = [0 if j == gi else -1 for j in range(len(options(q)))]
                    ev.append({"id": rid, "source": "e39_policy",
                               "questions": [{"qid": "q", "lab": lab, "words": word_ids(st, [it["gold_span"]])}]})
        i += 1
    return rows, ev


def check2_rows(k):
    rows = []
    if not (D38 / "check_train_v2.jsonl").exists():
        print("WARNING: no check_train_v2.jsonl yet (E38f); check2 stream empty", flush=True)
        return rows
    for r in map(json.loads, open(D38 / "check_train_v2.jsonl")):
        if r.get("source") != "e38_check_v2":
            continue                                   # v1 template checks are covered by the policy stream
        tg = {k: target(r["questions"][k], g["probabilities"]["true"] > 0.5)[0] for k, g in r["gold"].items()}
        rid = f"e39_check2/{r['id']}"
        rows.append(({"id": rid, "source": "e39_check2", "permissions": "libre", "state": r["state"],
                      "questions": r["questions"]}, {"id": rid, "q": tg, "k": k, "g": "e39_check2|" + r["id"].split("/")[0]}))
    return rows


def cf_rows(k):
    import cf_pairs
    from cf_pairs import load_edits, word_diff, changed_spans  # noqa: F401
    cf_pairs.D = D38
    pairs = [json.loads(l) for l in open(D38 / "cf2_pairs.jsonl")]
    rows1, edits1 = load_edits()                       # v1: valid pairs from cf_scored.jsonl
    for x in map(json.loads, open(D38 / "cf_scored.jsonl")):
        if x.get("valid"):
            r, e = rows1[x["id"]], edits1[x["id"]]
            qid, q = next(iter(r["questions"].items()))
            sa, sb = changed_spans(r["state"], e["edited_text"])
            pairs.append({"id": x["id"], "source": r["source"], "permissions": r["permissions"], "question_id": qid,
                          "question": q, "orig_text": r["state"], "edit_text": e["edited_text"], "gold": r["cf_gold"],
                          "target": r["cf_target"], "orig_spans": sa, "edit_spans": sb})
    rows, ev = [], []
    for p in pairs:
        q, qid = p["question"], p["question_id"]
        for side, text, ans, spans in (("orig", p["orig_text"], p["gold"], p["orig_spans"]),
                                       ("edit", p["edit_text"], p["target"], p["edit_spans"])):
            tg, gi = target(q, ans)
            rid = f"e39_cf/{p['id']}/{side}"
            rows.append(({"id": rid, "source": "e39_cf", "permissions": p.get("permissions", "libre"), "state": text,
                          "questions": {qid: q}}, {"id": rid, "q": {qid: tg}, "k": k, "g": f"e39_cf|{p['id']}"}))
            w = word_ids(text, spans)
            if w:
                lab = [0 if j == gi else -1 for j in range(len(options(q)))]
                ev.append({"id": rid, "source": "e39_cf", "questions": [{"qid": qid, "lab": lab, "words": w}]})
    return rows, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_policy", type=int, default=6000)
    ap.add_argument("--n_policy_ev", type=int, default=3000, help="cap on policy evidence entries")
    ap.add_argument("--k_policy", type=int, default=3, help="repeats of each policy row (about 5% of typed packs)")
    ap.add_argument("--k_check", type=int, default=3, help="repeats of each check2 row")
    ap.add_argument("--k_cf", type=int, default=4, help="repeats of each counterfactual row in the typed stream")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bench", required=True, help="E37 bench JSONL (13-gram decontamination; eval only, not shipped)")
    ap.add_argument("--policy_eval", default=None, help="policy_v0 eval set (default data/v3/e38/policy_v0.jsonl)")
    ap.add_argument("--data_root", default=None, help="default $HOROPTER_DATA or the current directory")
    ap.add_argument("--tokenizer", default="jhu-clsp/ettin-encoder-1b", help="length counts (the Ettin sizes share it)")
    a = ap.parse_args()
    global ROOT, SRC, OUT, D38, POLICY_EVAL
    if a.data_root:
        ROOT = Path(a.data_root).resolve()
        SRC, OUT, D38 = ROOT / "data/v3/prod", ROOT / "data/v3/prod2", ROOT / "data/v3/e38"
    POLICY_EVAL = Path(a.policy_eval) if a.policy_eval else D38 / "policy_v0.jsonl"
    if not POLICY_EVAL.exists():
        ap.error(f"{POLICY_EVAL} not found: make it with policy_gen.py --n 600 --seed 0 (its states are kept out)")
    rng = random.Random(a.seed)
    pol, pol_ev = policy_rows(a.n_policy, rng, a.k_policy)
    rng.shuffle(pol_ev)
    ck = check2_rows(a.k_check)
    cf, cf_ev = cf_rows(a.k_cf)
    bench = set()
    for r in map(json.loads, open(a.bench)):
        bench |= ngrams(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]))
    stats, keep = Counter(), []
    for row, tg in pol + ck + cf:
        if ngrams(row["state"]) & bench:
            stats[f"dropped_13gram_{row['source']}"] += 1
            continue
        keep.append((row, tg))
        stats[f"rows_{row['source']}"] += 1
        stats[f"decisions_{row['source']}"] += len(tg["q"]) * tg["k"]
    ids = {r["id"] for r, _ in keep}
    ev_new = [e for e in pol_ev[:a.n_policy_ev] + cf_ev if e["id"] in ids]
    stats["evidence_entries_new"] = len(ev_new)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targets").mkdir(exist_ok=True)
    (OUT / "lengths").mkdir(exist_ok=True)
    for sub in ("targets", "lengths"):                 # the E34 pools: links, not copies
        for f in (SRC / sub).iterdir():
            dst = OUT / sub / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
    pool = ROOT / "data/v3/e39/pool_e39.jsonl"
    pool.parent.mkdir(parents=True, exist_ok=True)
    with open(pool, "w") as f:
        for r, _ in keep:
            f.write(json.dumps(r) + "\n")
    with gzip.open(OUT / "targets/e39.jsonl.gz", "wt") as f:
        for _, t in keep:
            f.write(json.dumps(t) + "\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    s_len = [len(tok(r["state"], add_special_tokens=False)["input_ids"]) for r, _ in keep]
    q_len = [sum(len(tok(q["instructions"], add_special_tokens=False)["input_ids"]) + 4 * (1 + len(options(q)))
                 for q in r["questions"].values()) for r, _ in keep]
    json.dump({"ids": [r["id"] for r, _ in keep], "s": s_len, "q": q_len},
              gzip.open(OUT / "lengths/e39.json.gz", "wt"))
    stats["over_window_2048"] = sum(s > 2048 for s in s_len)
    with gzip.open(OUT / "evidence_prod2.jsonl.gz", "wt") as f:
        for line in gzip.open(SRC / "evidence_a25.jsonl.gz", "rt"):
            f.write(line)
        for e in ev_new:
            f.write(json.dumps(e) + "\n")
    man = json.load(open(SRC / "manifest.json"))
    man["pools"]["e39"] = str(pool.relative_to(ROOT))
    man["evidence"] = str((OUT / "evidence_prod2.jsonl.gz").relative_to(ROOT))
    args = {k: v for k, v in vars(a).items() if k in ("n_policy", "n_policy_ev", "k_policy", "k_check", "k_cf", "seed")}
    man["built_e39"] = {"by": "training/data/build_prod2.py", "args": args, "stats": dict(stats)}
    for f in ("eval_a25_dev.jsonl", "eval_a25_test.jsonl", "eval_a25_split.json"):
        dst = OUT / f
        if not dst.exists():
            os.symlink((SRC / f).resolve(), dst)
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    json.dump(dict(stats), open(OUT / "prod2_stats.json", "w"), indent=1)
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
