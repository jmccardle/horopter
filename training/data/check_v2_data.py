# SPDX-License-Identifier: MIT
"""E38f: data for check adapter v2. Targets E38e's failure ("rule 0 fires too easily": the trunk says a condition holds
when the case merely mentions its topic) with decoder-written policies and compiler-style conditions.

Qwen3-30B-A3B (self-hosted, via training/tools/llm_fanout.py) invents a policy (one of DOMAINS, rules with an exception and a numeric threshold) and
3 cases. For each case it writes ~6 conditions in the style of the E38e compiler ("the customer is a premium member"),
with true/false, at least half false and the false ones HARD NEGATIVES: about something the case explicitly mentions
(wrong value, wrong side of a threshold, negated, the exception that does not apply). A second, blind Qwen pass answers
each condition on the rendered state; only conditions where both passes agree are kept.

Each case is rendered two ways, as E38e's real items are: a JevBench-style document (policy + case prose) and a
JF100-style JSON state ({"policy", "case": {fields}}). No eval item is shown to the generator; states sharing a 13-gram
with the E37 bench are dropped anyway.

Paths: --data_dir (default $HOROPTER_DATA/data/v3/e38) holds the job / output files and check_train.jsonl (the v1
rows from node_data.py); assemble needs --bench (the E37 bench JSONL, eval only, never shipped) for the 13-gram check.

    python training/data/check_v2_data.py build [--n 600]   -> $D/ck2_jobs.jsonl
    python training/tools/llm_fanout.py run --endpoints endpoints.json --no-think --temperature 0.8 \
        --in $D/ck2_jobs.jsonl --out $D/ck2_gen.jsonl
    python training/data/check_v2_data.py verify            -> $D/ck2_verify_jobs.jsonl   (llm_fanout again -> ck2_verify.jsonl)
    python training/data/check_v2_data.py assemble --bench bench.jsonl
                                                 -> $D/check_train_v2.jsonl (v1 rows + v2 rows), $D/check_dev_v2.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cf_pairs import content  # noqa: E402

D = Path(os.environ.get("HOROPTER_DATA", ".")) / "data/v3/e38"
BENCH = None        # --bench: the E37 bench JSONL (13-gram decontamination; eval only, not shipped)
RESULTS = Path("results")
SOFT = 0.97
DOMAINS = ["retail returns", "employee leave", "expense reimbursement", "IT account access", "insurance claims",
           "library lending", "airline baggage", "hotel cancellation", "loan pre-approval", "school admissions",
           "warehouse safety", "robot charging procedure", "clinical trial eligibility", "visa applications",
           "software release approval", "apartment rental applications", "gym membership freezes",
           "grant funding eligibility", "data retention and deletion", "customer support escalation",
           "subscription refunds", "vehicle rental", "scholarship renewal", "food delivery compensation",
           "conference travel booking", "pharmacy prescription refills", "remote work requests",
           "bank overdraft fee waivers", "event ticket transfers", "manufacturing quality holds"]
STYLES = ["a terse internal policy with numbered rules", "a customer-facing FAQ paragraph", "a formal legal-style clause",
          "a standard operating procedure with steps", "an email from a manager announcing the rules"]

GEN_SCHEMA = {"type": "object", "properties": {
    "policy": {"type": "string"},
    "cases": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "properties": {
        "case_text": {"type": "string"},
        "case_fields": {"type": "object"},
        "checks": {"type": "array", "minItems": 5, "maxItems": 8, "items": {"type": "object", "properties": {
            "condition": {"type": "string"}, "holds": {"type": "boolean"},
            "kind": {"type": "string", "enum": ["plain", "near_miss", "exception", "threshold"]}},
            "required": ["condition", "holds", "kind"]}}},
        "required": ["case_text", "case_fields", "checks"]}}},
    "required": ["policy", "cases"]}
YN_SCHEMA = {"type": "object", "properties": {"holds": {"type": "boolean"}}, "required": ["holds"]}


def build(n, seed):
    rng = random.Random(seed)
    D.mkdir(parents=True, exist_ok=True)
    with open(D / "ck2_jobs.jsonl", "w") as f:
        for i in range(n):
            dom, sty = DOMAINS[i % len(DOMAINS)], rng.choice(STYLES)
            msg = (f"Invent a realistic policy about {dom}, written as {sty}. It has 4-7 rules, including at least one "
                   f"exception (a group or situation that is exempt from a rule or overrides it) and at least one "
                   f"numeric threshold (an amount, a number of days, a count).\n\n"
                   f"Then write 3 different cases. Each case is a short description of one person or request "
                   f"(case_text, 2-4 sentences) and the same facts as fields (case_fields: short snake_case keys; "
                   f"numbers, true/false or short strings).\n\n"
                   f"For each case, write 5-8 conditions in the style a rule compiler would write: a plain, "
                   f"self-contained statement about the case that is either true or false, e.g. 'the customer has "
                   f"been a member for more than 2 years' or 'the request was made at least 14 days in advance'. "
                   f"Give whether each condition holds for that case.\n"
                   f"- At least half of the conditions must be FALSE for the case.\n"
                   f"- The false ones must be hard: about something the case explicitly mentions, but with the wrong "
                   f"value, the wrong side of a threshold, the opposite status, or an exception that does not apply "
                   f"to this case (kind near_miss, threshold or exception).\n"
                   f"- Include at least one exception-type condition per case (true or false).\n"
                   f"- Every condition must be decidable from the case facts plus the policy text.")
            f.write(json.dumps({"id": f"ck2_{i:04d}", "messages": [{"role": "user", "content": msg}],
                                "params": {"response_format": {"type": "json_schema", "json_schema": {
                                    "name": "pol", "schema": GEN_SCHEMA}}}}) + "\n")
    print(f"{n} jobs -> {D / 'ck2_jobs.jsonl'}")


def states(g, j, c):
    """The two renderings of case j: document (policy + case prose) and JSON."""
    doc = f"{g['policy'].strip()}\n\nCase: {c['case_text'].strip()}"
    js = json.dumps({"policy": g["policy"].strip(), "case": c["case_fields"]})
    return {"doc": doc, "json": js}


def load_gen():
    out = {}
    for rec in map(json.loads, open(D / "ck2_gen.jsonl")):
        try:
            g = json.loads(content(rec))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(g.get("policy"), str) and len(g.get("cases") or []) >= 1:
            out[rec["id"]] = g
    return out


def verify():
    gen = load_gen()
    n = 0
    with open(D / "ck2_verify_jobs.jsonl", "w") as f:
        for gid, g in gen.items():
            for j, c in enumerate(g["cases"]):
                for reg, st in states(g, j, c).items():
                    for k, ch in enumerate(c["checks"]):
                        msg = (f"State:\n{st}\n\nDoes the case satisfy this condition: {ch['condition']}?\n"
                               f"Answer holds = true or false, using only the state.")
                        f.write(json.dumps({"id": f"{gid}::{j}::{reg}::{k}", "messages": [{"role": "user", "content": msg}],
                                            "params": {"response_format": {"type": "json_schema", "json_schema": {
                                                "name": "yn", "schema": YN_SCHEMA}}}}) + "\n")
                        n += 1
    print(f"{len(gen)} policies; {n} blind checks -> {D / 'ck2_verify_jobs.jsonl'}")


def ngrams(t, n=13):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def assemble(seed):
    gen = load_gen()
    blind = {}
    for rec in map(json.loads, open(D / "ck2_verify.jsonl")):
        try:
            blind[rec["id"]] = bool(json.loads(content(rec))["holds"])
        except Exception:  # noqa: BLE001
            pass
    bench = set()
    for r in map(json.loads, open(BENCH)):
        bench |= ngrams(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]))
    st_ = Counter()
    rows = []
    for gid, g in sorted(gen.items()):
        if ngrams(g["policy"]) & bench:
            st_["policy_dropped_13gram"] += 1
            continue
        held = int(gid.split("_")[1]) % 10 == 0      # 10% of policies held out as a dev set (not trained)
        for j, c in enumerate(g["cases"]):
            for reg, st in states(g, j, c).items():
                qs, gold = {}, {}
                for k, ch in enumerate(c["checks"]):
                    st_["checks"] += 1
                    b = blind.get(f"{gid}::{j}::{reg}::{k}")
                    if b is None or b != bool(ch["holds"]):
                        st_["disagree_or_missing"] += 1
                        continue
                    st_[f"kept_{'true' if ch['holds'] else 'false'}"] += 1
                    st_[f"kept_kind_{ch.get('kind')}"] += 1
                    qs[f"c{k}"] = {"type": "noul", "instructions": f"Does the case satisfy this condition: {ch['condition']}?"}
                    gold[f"c{k}"] = {"probabilities": {"true": SOFT if ch["holds"] else 1 - SOFT,
                                                       "false": 1 - SOFT if ch["holds"] else SOFT}}
                if qs:
                    rows.append({"id": f"{gid}/{j}/{reg}", "source": "e38_check_v2", "state": st, "questions": qs,
                                 "gold": gold, "dev": held})
    train = [r for r in rows if not r["dev"]]
    dev = [r for r in rows if r["dev"]]
    for r in rows:
        r.pop("dev")
    v1 = [json.loads(l) for l in open(D / "check_train.jsonl")]
    allr = v1 + train
    random.Random(seed).shuffle(allr)
    with open(D / "check_train_v2.jsonl", "w") as f:
        for r in allr:
            f.write(json.dumps(r) + "\n")
    with open(D / "check_dev_v2.jsonl", "w") as f:
        for r in dev:
            f.write(json.dumps(r) + "\n")
    res = dict(st_) | {"policies": len(gen), "v2_train_rows": len(train), "v2_dev_rows": len(dev), "v1_rows": len(v1),
                       "v2_train_questions": sum(len(r["questions"]) for r in train)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RESULTS / "check_v2_data.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "assemble"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--data_dir", default=None, help="default $HOROPTER_DATA/data/v3/e38")
    ap.add_argument("--bench", default=None, help="assemble: E37 bench JSONL for the 13-gram check (required)")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    global D, BENCH, RESULTS
    D, BENCH, RESULTS = Path(a.data_dir) if a.data_dir else D, a.bench, Path(a.results)
    if a.cmd == "assemble" and not BENCH:
        ap.error("assemble needs --bench (decontamination against the eval bench)")
    {"build": lambda: build(a.n, a.seed), "verify": verify, "assemble": lambda: assemble(a.seed)}[a.cmd]()


if __name__ == "__main__":
    main()
