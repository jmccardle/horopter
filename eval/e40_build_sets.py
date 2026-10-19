# SPDX-License-Identifier: MIT
"""E40: build real-world rule eval sets (ShARC, ConditionalQA, ProofWriter) in the typed wire format.

    python -I eval/e40_build_sets.py --datasets <dir> [--out $HOROPTER_DATA/data/v3/e40]
-> <out>/{sharc_dev,conditionalqa_dev,proofwriter_owa_d5_dev,proofwriter_cwa_d5_dev}.jsonl + build_stats.json

--datasets holds the downloaded source releases (not shipped; each under its own licence):
  sharc/sharc1-official/json/sharc_dev.json, conditionalqa/v1_0/{documents,dev}.json,
  proofwriter/proofwriter-dataset-V2020.12.3/{OWA,CWA}/depth-5/meta-dev.jsonl
These are dev splits, used for evaluation only.

Row: {"id", "set", "state", "questions": {qid: question}, "gold": {qid: answer}, "meta": {...}}.
Gold is "true"/"false" for noul, a criteria key for choice. Source files are read as JSON only (untrusted data).
"""
import argparse
import html
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DS = None   # --datasets (required)
OUT = Path(os.environ.get("HOROPTER_DATA", ".")) / "data/v3/e40"
SEED = 40
SHARC_MAX, CQA_MAX, PW_MAX = 6000, 8000, 6000
PW_PER_DEPTH = 300

SHARC_CRITERIA = {
    "yes": "The rule, applied to what the user has said, answers the user's question with yes.",
    "no": "The rule, applied to what the user has said, answers the user's question with no.",
    "irrelevant": "The rule text does not address the user's question at all.",
    "more_info": "The rule is relevant, but the user has not yet given enough information to answer; a follow-up question is needed.",
}
PW_OWA_CRITERIA = {
    "true": "The statement follows from the facts and rules.",
    "false": "The negation of the statement follows from the facts and rules.",
    "unknown": "Neither the statement nor its negation follows from the facts and rules.",
}


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sharc():
    src = json.load(open(DS / "sharc/sharc1-official/json/sharc_dev.json"))
    rows, skipped = [], 0
    for x in src:
        lines = [x["snippet"].strip(), "", "Scenario: " + (x["scenario"].strip() or "(none given)")]
        for h in x["history"]:
            lines.append(f"Q: {h['follow_up_question'].strip()} A: {h['follow_up_answer'].strip()}")
        lines.append("User question: " + x["question"].strip())
        state = "\n".join(lines)
        if len(state) > SHARC_MAX:
            skipped += 1
            continue
        a = x["answer"]
        gold = {"Yes": "yes", "No": "no", "Irrelevant": "irrelevant"}.get(a, "more_info")
        qs = {"q": {"type": "choice",
                    "instructions": "Given the rule and what the user has said, what is the answer to the user's question?",
                    "criteria": SHARC_CRITERIA}}
        g = {"q": gold}
        binary = gold in ("yes", "no")
        if binary:   # binary-only subset: the user question itself as a noul
            qs["qb"] = {"type": "noul", "instructions": x["question"].strip()}
            g["qb"] = "true" if gold == "yes" else "false"
        rows.append({"id": f"sharc-{x['utterance_id']}", "set": "sharc_dev", "state": state, "questions": qs,
                     "gold": g, "meta": {"depth": len(x["history"]), "binary": binary, "tree_id": x["tree_id"],
                                         "follow_up": a if gold == "more_info" else None,
                                         "has_scenario": bool(x["scenario"].strip()), "source_url": x["source_url"]}})
    write("sharc_dev", rows)
    return {"rows": len(rows), "skipped_long": skipped, "source": len(src),
            "gold": dict(Counter(r["gold"]["q"] for r in rows)),
            "depth": dict(sorted(Counter(r["meta"]["depth"] for r in rows).items()))}


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s)).strip()


def cqa():
    docs = {d["url"]: d for d in json.load(open(DS / "conditionalqa/v1_0/documents.json"))}
    src = json.load(open(DS / "conditionalqa/v1_0/dev.json"))
    rows, st = [], Counter()
    for x in src:
        if x["not_answerable"]:
            st["skipped_not_answerable"] += 1
            continue
        ans = {a[0].strip().lower() for a in x["answers"]}
        if not ans or not ans <= {"yes", "no"}:
            st["skipped_not_yesno"] += 1
            continue
        if len(ans) > 1:   # yes under some conditions, no under others: no single gold
            st["skipped_mixed_yes_no"] += 1
            continue
        d = docs.get(x["url"])
        if d is None:
            st["skipped_no_document"] += 1
            continue
        text = "\n".join(t for t in (strip_html(c) for c in d["contents"]) if t)
        state = d["title"] + "\n" + text + "\n\nScenario: " + x["scenario"].strip()
        if len(state) > CQA_MAX:
            st["skipped_long"] += 1
            continue
        conds = any(a[1] for a in x["answers"])
        g = ans.pop()
        rows.append({"id": f"cqa-{x['id']}", "set": "conditionalqa_dev", "state": state,
                     "questions": {"q": {"type": "noul", "instructions": x["question"].strip()}},
                     "gold": {"q": "true" if g == "yes" else "false"},
                     "meta": {"has_conditions": conds, "url": x["url"], "state_chars": len(state)}})
    write("conditionalqa_dev", rows)
    return {"rows": len(rows), "source": len(src), **st,
            "gold": dict(Counter(r["gold"]["q"] for r in rows)),
            "has_conditions": dict(Counter(r["meta"]["has_conditions"] for r in rows))}


def proofwriter(world):
    path = DS / f"proofwriter/proofwriter-dataset-V2020.12.3/{world}/depth-5/meta-dev.jsonl"
    theories, by_depth, st = {}, defaultdict(list), Counter()
    for line in open(path):
        t = json.loads(line)
        if len(t["theory"]) > PW_MAX:
            st["skipped_long_theories"] += 1
            continue
        theories[t["id"]] = t
        for qid, q in t["questions"].items():
            if 0 <= q["QDep"] <= 5:
                by_depth[q["QDep"]].append((t["id"], qid))
            else:
                st["skipped_depth_gt5"] += 1
    rng = random.Random(SEED)
    picked = defaultdict(list)   # theory id -> [qid]
    for dep in range(6):
        pool = sorted(by_depth[dep])
        for tid, qid in rng.sample(pool, min(PW_PER_DEPTH, len(pool))):
            picked[tid].append(qid)
    rows = []
    name = f"proofwriter_{world.lower()}_d5_dev"
    for tid in sorted(picked):   # one row per theory: all its sampled questions go in one request
        t = theories[tid]
        qs, g, deps = {}, {}, {}
        for qid in sorted(picked[tid], key=lambda s: int(s[1:])):
            q = t["questions"][qid]
            stmt = q["question"].strip().rstrip(".")
            text = f"Is it true that {stmt[0].lower() + stmt[1:] if stmt.split()[0] in ('The',) else stmt}?"
            ans = str(q["answer"]).lower()
            if world == "OWA":
                qs[qid] = {"type": "choice", "instructions": text, "criteria": PW_OWA_CRITERIA}
            else:
                qs[qid] = {"type": "noul", "instructions": text}
            g[qid] = ans
            deps[qid] = q["QDep"]
        rows.append({"id": f"pw-{tid}", "set": name, "state": t["theory"], "questions": qs, "gold": g,
                     "meta": {"depth": deps, "maxD": t["maxD"], "NFact": t["NFact"], "NRule": t["NRule"]}})
    write(name, rows)
    gc = Counter((r["meta"]["depth"][q], r["gold"][q]) for r in rows for q in r["questions"])
    return {"rows": len(rows), "questions": sum(len(r["questions"]) for r in rows), **st,
            "per_depth_gold": {d: {k: gc[(d, k)] for k in ("true", "false", "unknown") if gc[(d, k)]} for d in range(6)}}


def main():
    global DS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="dir with the downloaded ShARC / ConditionalQA / ProofWriter files")
    ap.add_argument("--out", default=None, help="default $HOROPTER_DATA/data/v3/e40")
    a = ap.parse_args()
    DS, OUT = Path(a.datasets), Path(a.out) if a.out else OUT
    stats = {"sharc_dev": sharc(), "conditionalqa_dev": cqa(),
             "proofwriter_owa_d5_dev": proofwriter("OWA"), "proofwriter_cwa_d5_dev": proofwriter("CWA")}
    json.dump(stats, open(OUT / "build_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
