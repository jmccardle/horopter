# SPDX-License-Identifier: MIT
"""E38c: counterfactual minimal pairs (beta-2 "cf" stream): an LLM edits a text minimally so the answer flips, a second
blind LLM pass verifies the flip. LLM jobs run through training/tools/llm_fanout.py (any OpenAI-compatible server).

  build   : sample typed rows (libre, one question, text state 200-1500 chars, gold known, <= 25 per source) from the
            a23 pool; one LLM job each: "make the smallest edit to the text so the answer becomes <target>".
  verify  : parse the edits; word-level diff size; a second LLM job per edit answers the question on the edited
            text blind (judge check that the label really flipped).
  score   : judge flip rate, edit size, and how often the live trunk already follows the edit (asked on both
            texts, through a running server); the changed words are the would-be span labels.
  v2 (build2 / verify2 / select2 / score2): see below; build_prod2.py reads cf_scored.jsonl (v1) and cf2_pairs.jsonl (v2).

Paths: --data_dir (default $HOROPTER_DATA/data/v3/e38) holds every job / output file; --pool is the typed pool to
sample from (default $HOROPTER_DATA/data/v3/e09/pool_a23.jsonl); --results for the summary JSON (default results/).

    python training/data/cf_pairs.py build  --n 600
    python training/tools/llm_fanout.py run --endpoints endpoints.json --in $D/cf_jobs.jsonl --out $D/cf_edits.jsonl
    python training/data/cf_pairs.py verify
    python training/tools/llm_fanout.py run --endpoints endpoints.json --in $D/cf_check_jobs_v2.jsonl --out $D/cf_checks_v3.jsonl
    python training/data/cf_pairs.py score            # needs a server (eval/client.py, HOROPTER_URL)
    python training/data/cf_pairs.py build2 | verify2 | select2 | score2   (same pattern, cf2_* files)
"""
import argparse
import difflib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("HOROPTER_DATA", "."))
D = DATA_ROOT / "data/v3/e38"
POOL = DATA_ROOT / "data/v3/e09/pool_a23.jsonl"
RESULTS = Path("results")
MODEL = "horopter"   # --model: server model id (the research runs asked the routed beta)

EDIT_SCHEMA = {"type": "object", "properties": {"edited_text": {"type": "string"}, "changed": {"type": "string"}},
               "required": ["edited_text", "changed"]}
ANS_SCHEMA = lambda opts: {"type": "object", "properties": {"answer": {"type": "string", "enum": opts}},  # noqa: E731
                           "required": ["answer"]}


def options(q):
    return ["true", "false"] if q["type"] == "noul" else [str(k) for k in q["criteria"]]


def gold_name(q, g):
    if q["type"] == "noul":
        return "true" if g in (True, "true", "True") else "false"
    return str(g)


def describe(q):
    if q["type"] == "noul":
        return f"Question (yes/no): {q['instructions']}\nOptions: true / false"
    crit = q["criteria"]
    lines = [f"- {k}: {v}" for k, v in crit.items()] if isinstance(crit, dict) else \
        [f"- {i}: {v}" for i, v in enumerate(crit)]
    return f"Question: {q['instructions']}\nOptions:\n" + "\n".join(lines)


def build(n, seed):
    rng = random.Random(seed)
    per = defaultdict(list)
    for l in open(POOL):
        r = json.loads(l)
        if r.get("permissions") != "libre" or not isinstance(r["state"], str) or not 200 <= len(r["state"]) <= 1500:
            continue
        if len(r["questions"]) != 1:
            continue
        qid, q = next(iter(r["questions"].items()))
        g = (r.get("gold") or {}).get(qid)
        if g is None or q["type"] not in ("noul", "choice"):
            continue
        per[r["source"]].append(r)
    picks = []
    for s, rs in sorted(per.items()):
        picks += rng.sample(rs, min(25, len(rs)))
    rng.shuffle(picks)
    picks = picks[:n]
    D.mkdir(parents=True, exist_ok=True)
    with open(D / "cf_rows.jsonl", "w") as fr, open(D / "cf_jobs.jsonl", "w") as fj:
        for r in picks:
            qid, q = next(iter(r["questions"].items()))
            g = gold_name(q, r["gold"][qid])
            target = rng.choice([o for o in options(q) if o != g])
            fr.write(json.dumps({**r, "cf_gold": g, "cf_target": target}) + "\n")
            msg = (f"Text:\n{r['state']}\n\n{describe(q)}\n\nThe correct answer for this text is: {g}.\n\n"
                   f"Make the SMALLEST possible edit to the text (change, add or delete as few words as possible, "
                   f"keep everything else identical) so that the correct answer becomes: {target}. "
                   f"Return the full edited text and a short note of what you changed.")
            fj.write(json.dumps({"id": r["id"], "messages": [{"role": "user", "content": msg}],
                                 "params": {"response_format": {"type": "json_schema",
                                                                "json_schema": {"name": "edit", "schema": EDIT_SCHEMA}}}})
                     + "\n")
    print(f"{len(picks)} rows from {len(per)} sources -> {D / 'cf_jobs.jsonl'}")


def content(rec):
    for k in ("content", "text", "output"):
        if isinstance(rec.get(k), str):
            return rec[k]
    try:
        return rec["response"]["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return None


def word_diff(a, b):
    wa, wb = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return changed, len(wa)


def load_edits():
    rows = {r["id"]: r for r in map(json.loads, open(D / "cf_rows.jsonl"))}
    edits = {}
    for rec in map(json.loads, open(D / "cf_edits.jsonl")):
        c = content(rec)
        if not rec.get("ok", True) or c is None:
            continue
        try:
            e = json.loads(c)
        except Exception:  # noqa: BLE001
            continue
        if rec["id"] in rows and isinstance(e.get("edited_text"), str):
            edits[rec["id"]] = e
    return rows, edits


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHECKS = "cf_checks_v3.jsonl"   # v2 (gemma, 20-token cap) was truncated by its reasoning output; v3 = Qwen3-30B-A3B, 400 tokens


def lettered(q):
    """Options as letters (v1 asked for the full option text under a 40-token cap: long options were cut off)."""
    opts = options(q)
    if q["type"] == "noul":
        body = f"Question (yes/no): {q['instructions']}\nOptions:\nA: true\nB: false"
    else:
        crit = q["criteria"]
        desc = (lambda o: crit[o]) if isinstance(crit, dict) else (lambda o: o)
        body = f"Question: {q['instructions']}\nOptions:\n" + "\n".join(
            f"{LETTERS[i]}: {o}" + (f" ({desc(o)})" if isinstance(crit, dict) and desc(o) != o else "")
            for i, o in enumerate(opts))
    return body, {LETTERS[i]: o for i, o in enumerate(opts)}


def verify():
    rows, edits = load_edits()
    with open(D / "cf_check_jobs_v2.jsonl", "w") as f:
        for rid, e in edits.items():
            r = rows[rid]
            q = next(iter(r["questions"].values()))
            body, lm = lettered(q)
            for tag, text in (("orig", r["state"]), ("edit", e["edited_text"])):
                msg = f"Text:\n{text}\n\n{body}\n\nAnswer with the letter of the single option the text supports."
                f.write(json.dumps({"id": f"{rid}::{tag}", "messages": [{"role": "user", "content": msg}],
                                    "params": {"response_format": {"type": "json_schema", "json_schema": {
                                        "name": "ans", "schema": ANS_SCHEMA(list(lm))}}}}) + "\n")
    print(f"{len(edits)} parsed edits / {len(rows)} rows -> {D / 'cf_check_jobs_v2.jsonl'}")


def _ask():
    sys.path.insert(0, str(HERE.parents[1] / "eval"))
    from client import ask
    return ask


def score():
    ask = _ask()
    rows, edits = load_edits()
    judge = {}
    for rec in map(json.loads, open(D / CHECKS)):
        try:
            rid = rec["id"].split("::")[0]
            judge[rec["id"]] = lettered(next(iter(rows[rid]["questions"].values())))[1][json.loads(content(rec))["answer"]]
        except Exception:  # noqa: BLE001
            pass
    out, by_src = [], defaultdict(list)
    for rid, e in edits.items():
        r = rows[rid]
        qid, q = next(iter(r["questions"].items()))
        nchg, nw = word_diff(r["state"], e["edited_text"])
        rec = {"id": rid, "source": r["source"], "words_changed": nchg, "words": nw,
               "judge_orig": judge.get(f"{rid}::orig"), "judge_edit": judge.get(f"{rid}::edit"),
               "gold": r["cf_gold"], "target": r["cf_target"]}
        rec["valid"] = rec["judge_orig"] == r["cf_gold"] and rec["judge_edit"] == r["cf_target"] and 0 < nchg <= 12
        try:
            preds = []
            for text in (r["state"], e["edited_text"]):
                o, _ = ask(text, {qid: q}, MODEL)
                a = o["answers"][qid]
                preds.append(("true" if a["noul"] >= 0.5 else "false") if q["type"] == "noul"
                             else max(a["probabilities"], key=a["probabilities"].get))
            rec["trunk_orig"], rec["trunk_edit"] = preds
        except Exception as ex:  # noqa: BLE001
            rec["trunk_error"] = str(ex)[:200]
        out.append(rec)
        by_src[r["source"]].append(rec)
    with open(D / "cf_scored.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    v = [x for x in out if x["valid"]]
    both = [x for x in v if "trunk_orig" in x]
    res = {"rows": len(rows), "parsed_edits": len(edits), "judge_valid": len(v),
           "valid_rate": round(len(v) / max(1, len(rows)), 3),
           "median_words_changed_valid": sorted(x["words_changed"] for x in v)[len(v) // 2] if v else None,
           "judge_agrees_orig_gold": round(sum(x["judge_orig"] == x["gold"] for x in out) / max(1, len(out)), 3),
           "trunk_right_on_both": round(sum(x["trunk_orig"] == x["gold"] and x["trunk_edit"] == x["target"]
                                            for x in both) / max(1, len(both)), 3),
           "trunk_right_orig_only": round(sum(x["trunk_orig"] == x["gold"] and x["trunk_edit"] != x["target"]
                                              for x in both) / max(1, len(both)), 3),
           "valid_by_source": {s: f"{sum(x['valid'] for x in xs)}/{len(xs)}" for s, xs in sorted(by_src.items())}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RESULTS / "cf_pilot.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# ---- v2 (2026-10-18): NLI / fact / short-classification sources only (the v1 yield table), Qwen3-30B-A3B as the
# editor proposing 3 alternative edits, each checked blind; the shortest verified edit is kept.
#   build2 -> llm_fanout (cf2_jobs -> cf2_edits) -> verify2 -> llm_fanout (cf2_check_jobs -> cf2_checks) -> select2 [-> score2: needs a server]
V2_SOURCES = ["nli_wanli", "nli_vitaminc", "nli_fever", "mnli", "snli", "fever", "boolq", "dbpedia", "banking77"]
EDITS3_SCHEMA = {"type": "object", "properties": {"edits": {
    "type": "array", "minItems": 3, "maxItems": 3, "items": EDIT_SCHEMA}}, "required": ["edits"]}


def build2(per, seed):
    rng = random.Random(seed)
    used = {r["id"] for r in map(json.loads, open(D / "cf_rows.jsonl"))}   # keep v1's rows out
    per_src = defaultdict(list)
    for l in open(POOL):
        r = json.loads(l)
        if r["source"] not in V2_SOURCES or r["id"] in used or r.get("permissions") != "libre":
            continue
        if not isinstance(r["state"], str) or not 200 <= len(r["state"]) <= 1500 or len(r["questions"]) != 1:
            continue
        qid, q = next(iter(r["questions"].items()))
        if (r.get("gold") or {}).get(qid) is None or q["type"] not in ("noul", "choice"):
            continue
        per_src[r["source"]].append(r)
    picks = []
    for s in V2_SOURCES:
        picks += rng.sample(per_src[s], min(per, len(per_src[s])))
    rng.shuffle(picks)
    with open(D / "cf2_rows.jsonl", "w") as fr, open(D / "cf2_jobs.jsonl", "w") as fj:
        for r in picks:
            qid, q = next(iter(r["questions"].items()))
            g = gold_name(q, r["gold"][qid])
            target = rng.choice([o for o in options(q) if o != g])
            fr.write(json.dumps({**r, "cf_gold": g, "cf_target": target}) + "\n")
            msg = (f"Text:\n{r['state']}\n\n{describe(q)}\n\nThe correct answer for this text is: {g}.\n\n"
                   f"Propose THREE different minimal edits. Each edit changes, adds or deletes as few words as possible "
                   f"(ideally 1-5), keeps everything else identical, and makes the correct answer become: {target}. "
                   f"The edited text must clearly support {target}, not merely become ambiguous. For each edit, return "
                   f"the full edited text and a short note of what you changed.")
            fj.write(json.dumps({"id": r["id"], "messages": [{"role": "user", "content": msg}],
                                 "params": {"response_format": {"type": "json_schema",
                                                                "json_schema": {"name": "edits", "schema": EDITS3_SCHEMA}}}})
                     + "\n")
    print(f"{len(picks)} rows ({', '.join(f'{s} {min(per, len(per_src[s]))}' for s in V2_SOURCES)}) -> {D / 'cf2_jobs.jsonl'}")


def load_edits2():
    rows = {r["id"]: r for r in map(json.loads, open(D / "cf2_rows.jsonl"))}
    edits = {}
    for rec in map(json.loads, open(D / "cf2_edits.jsonl")):
        c = content(rec)
        if not rec.get("ok", True) or c is None or rec["id"] not in rows:
            continue
        try:
            es = json.loads(c)["edits"]
        except Exception:  # noqa: BLE001
            continue
        seen, keep = set(), []
        for e in es:
            t = e.get("edited_text") if isinstance(e, dict) else None
            if isinstance(t, str) and t.strip() and t != rows[rec["id"]]["state"] and t not in seen:
                seen.add(t)
                keep.append(e)
        if keep:
            edits[rec["id"]] = keep
    return rows, edits


def verify2():
    rows, edits = load_edits2()
    n = 0
    with open(D / "cf2_check_jobs.jsonl", "w") as f:
        for rid, es in edits.items():
            q = next(iter(rows[rid]["questions"].values()))
            body, lm = lettered(q)
            for tag, text in [("orig", rows[rid]["state"])] + [(f"edit{i}", e["edited_text"]) for i, e in enumerate(es)]:
                msg = f"Text:\n{text}\n\n{body}\n\nAnswer with the letter of the single option the text supports."
                f.write(json.dumps({"id": f"{rid}::{tag}", "messages": [{"role": "user", "content": msg}],
                                    "params": {"response_format": {"type": "json_schema", "json_schema": {
                                        "name": "ans", "schema": ANS_SCHEMA(list(lm))}}}}) + "\n")
                n += 1
    print(f"{len(edits)} rows with edits ({sum(map(len, edits.values()))} edits) / {len(rows)} rows; {n} checks "
          f"-> {D / 'cf2_check_jobs.jsonl'}")


def changed_spans(a, b):
    """Char spans of the changed words: (spans in a, spans in b), from a word-level diff."""
    import re
    ta = [(m.start(), m.end()) for m in re.finditer(r"\S+", a)]
    tb = [(m.start(), m.end()) for m in re.finditer(r"\S+", b)]
    sm = difflib.SequenceMatcher(a=[a[s:e] for s, e in ta], b=[b[s:e] for s, e in tb], autojunk=False)
    sa, sb = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            if i2 > i1:
                sa.append([ta[i1][0], ta[i2 - 1][1]])
            if j2 > j1:
                sb.append([tb[j1][0], tb[j2 - 1][1]])
    return sa, sb


def select2():
    rows, edits = load_edits2()
    judge = {}
    for rec in map(json.loads, open(D / "cf2_checks.jsonl")):
        try:
            rid = rec["id"].split("::")[0]
            judge[rec["id"]] = lettered(next(iter(rows[rid]["questions"].values())))[1][json.loads(content(rec))["answer"]]
        except Exception:  # noqa: BLE001
            pass
    pairs, by_src = [], defaultdict(lambda: Counter())
    n_props = n_props_ok = 0
    for rid, r in rows.items():
        src = by_src[r["source"]]
        src["rows"] += 1
        es = edits.get(rid, [])
        src["parsed"] += bool(es)
        orig_ok = judge.get(f"{rid}::orig") == r["cf_gold"]
        src["orig_ok"] += orig_ok
        good = []
        for i, e in enumerate(es):
            n_props += 1
            nchg, nw = word_diff(r["state"], e["edited_text"])
            ok = orig_ok and judge.get(f"{rid}::edit{i}") == r["cf_target"] and 0 < nchg <= 12
            n_props_ok += ok
            src["edit_flips"] += judge.get(f"{rid}::edit{i}") == r["cf_target"]
            src["proposals"] += 1
            if ok:
                good.append((nchg, i, e))
        if good:
            nchg, i, e = min(good, key=lambda x: (x[0], x[1]))
            sa, sb = changed_spans(r["state"], e["edited_text"])
            qid, q = next(iter(r["questions"].items()))
            pairs.append({"id": rid, "source": r["source"], "permissions": r["permissions"], "question_id": qid,
                          "question": q, "orig_text": r["state"], "edit_text": e["edited_text"], "gold": r["cf_gold"],
                          "target": r["cf_target"], "words_changed": nchg, "proposal": i, "n_verified": len(good),
                          "orig_spans": sa, "edit_spans": sb, "note": e.get("changed")})
            src["valid"] += 1
    with open(D / "cf2_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    wc = sorted(p["words_changed"] for p in pairs)
    res = {"rows": len(rows), "rows_with_edits": len(edits), "proposals": n_props, "proposals_verified": n_props_ok,
           "valid_pairs": len(pairs), "valid_rate_rows": round(len(pairs) / max(1, len(rows)), 3),
           "proposal_verified_rate": round(n_props_ok / max(1, n_props), 3),
           "median_words_changed": wc[len(wc) // 2] if wc else None,
           "pairs_from_first_proposal_only": sum(p["proposal"] == 0 for p in pairs),
           "by_source": {s: dict(c) | {"valid_rate": round(c["valid"] / max(1, c["rows"]), 3)}
                         for s, c in sorted(by_src.items())}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RESULTS / "cf_v2.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def score2():
    """Trunk on both sides of each v2 pair (needs a running server)."""
    ask = _ask()
    pairs = [json.loads(l) for l in open(D / "cf2_pairs.jsonl")]
    both = orig_only = n = 0
    for p in pairs:
        q, qid = p["question"], p["question_id"]
        preds = []
        for text in (p["orig_text"], p["edit_text"]):
            a = ask(text, {qid: q}, MODEL)[0]["answers"][qid]
            preds.append(("true" if a["noul"] >= 0.5 else "false") if q["type"] == "noul"
                         else max(a["probabilities"], key=a["probabilities"].get))
        n += 1
        both += preds[0] == p["gold"] and preds[1] == p["target"]
        orig_only += preds[0] == p["gold"] and preds[1] != p["target"]
    res = json.load(open(RESULTS / "cf_v2.json"))
    res["trunk"] = {"pairs": n, "right_on_both": round(both / max(1, n), 3), "right_orig_only": round(orig_only / max(1, n), 3)}
    json.dump(res, open(RESULTS / "cf_v2.json", "w"), indent=1)
    print(res["trunk"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "score", "build2", "verify2", "select2", "score2"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--per", type=int, default=70, help="build2: rows per source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default=None, help="job / output files (default $HOROPTER_DATA/data/v3/e38)")
    ap.add_argument("--pool", default=None, help="typed pool to sample (default $HOROPTER_DATA/data/v3/e09/pool_a23.jsonl)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="horopter", help="score / score2: server model id")
    a = ap.parse_args()
    global D, POOL, RESULTS, MODEL
    D, POOL = Path(a.data_dir) if a.data_dir else D, Path(a.pool) if a.pool else POOL
    RESULTS, MODEL = Path(a.results), a.model
    {"build": lambda: build(a.n, a.seed), "verify": verify, "score": score, "build2": lambda: build2(a.per, a.seed + 1),
     "verify2": verify2, "select2": select2, "score2": score2}[a.cmd]()


if __name__ == "__main__":
    main()
