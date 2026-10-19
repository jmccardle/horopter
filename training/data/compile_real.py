# SPDX-License-Identifier: MIT
"""E38e: the interpreter on REAL policy items. A decoder LLM (self-hosted) compiles each policy into a decision list offline;
the trunk + check adapter answers the conditions; the answer is executed in code.

Items: JF100 policy_rules, formal_logic, evidence_integration (JSON states) and JevBench hard long_policy (text).
Program = ordered decision list {"rules": [{"condition", "outcome"}], "default"}: the first rule whose condition holds
gives the outcome. Any rule set with precedence compiles to one.
  - JF100 states keep the case in its own field, so the compiler sees the policy, question, options and case FIELD
    NAMES, but no case values: a case-independent program, compiled once per policy ("offline, cached").
  - JevBench long_policy mixes the policy and the case in one document, so its compiler sees the whole state. The
    decoder's own direct answer is asked too, to see whether the trunk adds anything over the decoder.

The items come from the E37 bench (--bench, eval only, never shipped); this is an evaluation, not a data builder.

    python training/data/compile_real.py build --bench bench.jsonl      -> $D/real_jobs.jsonl (program + direct jobs)
    python training/tools/llm_fanout.py run --endpoints endpoints.json --in $D/real_jobs.jsonl --out $D/real_out.jsonl
    python training/data/compile_real.py run --bench bench.jsonl [--node_model horopter-check]
                                                                        -> results/real_<tag>.{json,md}
($D = --data_dir, default $HOROPTER_DATA/data/v3/e38; the server URL comes from HOROPTER_URL, see eval/client.py.)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "eval"))
D = Path(os.environ.get("HOROPTER_DATA", ".")) / "data/v3/e38"
BENCH = None        # --bench (required)
RESULTS = Path("results")
FAMS = {("jf100", "policy_rules"), ("jf100", "formal_logic"), ("jf100", "evidence_integration"),
        ("jevbench_hard", "long_policy")}


def items():
    return [r for r in map(json.loads, open(BENCH)) if (r["set"], r.get("family")) in FAMS
            and r["questions"]["q"]["type"] == "choice"]   # decision lists map to option keys


def by_name(q):
    """JF100 options are letters (A-D) whose order differs between twins: programs then name the option VALUE, which
    is mapped back to this item's letter at run time (v1 shared letter outcomes across twins: wrong for one twin)."""
    return all(len(k) == 1 for k in q["criteria"])


def opts_text(q):
    if by_name(q):
        return "\n".join(f"- {v}" for v in q["criteria"].values())
    return "\n".join(f"- {k}: {v}" for k, v in q["criteria"].items())


def program_schema(keys):
    return {"type": "object", "properties": {
        "rules": {"type": "array", "items": {"type": "object", "properties": {
            "condition": {"type": "string"}, "outcome": {"type": "string", "enum": keys}},
            "required": ["condition", "outcome"]}},
        "default": {"type": "string", "enum": keys}}, "required": ["rules", "default"]}


def build():
    D.mkdir(parents=True, exist_ok=True)
    seen_prog = set()
    with open(D / "real_jobs.jsonl", "w") as f:
        for r in items():
            q = r["questions"]["q"]
            keys = list(q["criteria"].values()) if by_name(q) else list(q["criteria"])
            st = r["state"]
            if isinstance(st, dict) and "policy" in st:
                fields = sorted((st.get("case") or {k: v for k, v in st.items() if k != "policy"}).keys())
                src = (f"Policy:\n{st['policy']}\n\nThe case will be given as fields: {', '.join(fields)} "
                       f"(values not shown).")
                pkey = json.dumps([st["policy"], q["instructions"], sorted(q["criteria"].values()), fields])
            else:
                src = f"Document:\n{st if isinstance(st, str) else json.dumps(st)}"
                pkey = r["id"]
            task = (f"{src}\n\nQuestion: {q['instructions']}\nOptions:\n{opts_text(q)}\n\n"
                    "Compile the rules into an ordered decision list a checker will execute on the case: each rule is a "
                    "condition about the case's facts, written as a plain self-contained statement that is either true or "
                    "false for a case (for example 'the item was bought within 14 days'), and the option it leads to. "
                    "The FIRST rule whose condition is true decides; if none is true, the default applies. Respect the "
                    "policy's precedence (exceptions and checks that come first go first). Do not decide the case yourself.")
            if pkey not in seen_prog:   # one program per distinct policy (JF100 twins share theirs)
                seen_prog.add(pkey)
                f.write(json.dumps({"id": f"prog{'2' if by_name(q) else ''}::{r['id']}", "messages": [{"role": "user", "content": task}],
                                    "params": {"response_format": {"type": "json_schema", "json_schema": {
                                        "name": "program", "schema": program_schema(keys)}}}}) + "\n")
            direct = (f"{'Document' if isinstance(st, str) else 'State'}:\n{st if isinstance(st, str) else json.dumps(st)}"
                      f"\n\nQuestion: {q['instructions']}\nOptions:\n{opts_text(q)}\n\nAnswer with the option key.")
            f.write(json.dumps({"id": f"direct{'2' if by_name(q) else ''}::{r['id']}", "messages": [{"role": "user", "content": direct}],
                                "params": {"response_format": {"type": "json_schema", "json_schema": {
                                    "name": "ans", "schema": {"type": "object", "properties": {
                                        "answer": {"type": "string", "enum": keys}}, "required": ["answer"]}}}}})
                    + "\n")
    print(f"{sum(1 for _ in open(D / 'real_jobs.jsonl'))} jobs -> {D / 'real_jobs.jsonl'}")


def program_key(r):
    st, q = r["state"], r["questions"]["q"]
    if isinstance(st, dict) and "policy" in st:
        fields = sorted((st.get("case") or {k: v for k, v in st.items() if k != "policy"}).keys())
        return json.dumps([st["policy"], q["instructions"], sorted(q["criteria"].values()), fields])
    return r["id"]


def run(node_model, tag, oneshot_model="horopter"):
    from client import ask
    from cf_pairs import content
    out = {}
    for rec in map(json.loads, open(D / "real_out.jsonl")):
        try:
            out[rec["id"]] = json.loads(content(rec))
        except Exception:  # noqa: BLE001
            pass
    rows = items()
    # programs keyed by policy: twins without their own job use their partner's
    prog_by_key = {}
    for r in rows:
        k = f"prog{'2' if by_name(r['questions']['q']) else ''}::{r['id']}"
        if k in out:
            prog_by_key[program_key(r)] = out[k]
    res, recs = defaultdict(lambda: defaultdict(list)), []
    for r in rows:
        q = r["questions"]["q"]
        g = str(r["gold"]["q"])
        prog = prog_by_key.get(program_key(r))
        name2key = {v: k for k, v in q["criteria"].items()} if by_name(q) else {k: k for k in q["criteria"]}
        direct = name2key.get((out.get(f"direct{'2' if by_name(q) else ''}::{r['id']}") or {}).get("answer"))
        rec = {"id": r["id"], "gold": g, "direct": direct, "n_rules": len(prog["rules"]) if prog else None}
        if prog and prog["rules"]:
            qs = {f"c{i}": {"type": "noul", "instructions": f"Does the case satisfy this condition: {x['condition']}?"}
                  for i, x in enumerate(prog["rules"])}
            o, ms = ask(r["state"], qs, node_model)
            ps = [o["answers"][f"c{i}"]["noul"] for i in range(len(prog["rules"]))]
            fired = next((i for i, p in enumerate(ps) if p >= 0.5), None)
            rec.update(checks=[round(p, 3) for p in ps], fired=fired, ms=round(ms, 1),
                       interp=name2key.get(prog["rules"][fired]["outcome"] if fired is not None else prog["default"]))
        elif prog:
            rec["interp"] = name2key.get(prog["default"])
        o, _ = ask(r["state"], {"q": q}, oneshot_model)
        rec["oneshot"] = max(o["answers"]["q"]["probabilities"], key=o["answers"]["q"]["probabilities"].get)
        recs.append(rec)
        fam = f"{r['set']}/{r['family']}"
        for k in ("oneshot", "direct", "interp"):
            res[fam][k].append(rec.get(k) == g)
            res["all"][k].append(rec.get(k) == g)
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"raw/real_{tag}.jsonl", "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    tab = {fam: {k: round(100 * sum(v) / len(v), 1) for k, v in d.items()} | {"n": len(d["oneshot"])}
           for fam, d in res.items()}
    json.dump(tab, open(RESULTS / f"real_{tag}.json", "w"), indent=1)
    lines = [f"# E38e real policy items: decoder-compiled decision list + trunk checks ({node_model})", "",
             "| family | n | trunk one-shot | decoder direct | interpreter (decoder program + trunk checks) |",
             "|---|---|---|---|---|"]
    for fam in sorted(tab, key=lambda k: (k == "all", k)):
        d = tab[fam]
        lines.append(f"| {fam} | {d['n']} | {d['oneshot']} | {d['direct']} | {d['interp']} |")
    open(RESULTS / f"real_{tag}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--node_model", default="horopter-check")
    ap.add_argument("--tag", default="check")
    ap.add_argument("--oneshot_model", default="horopter", help="trunk only: horopter-base")
    ap.add_argument("--bench", required=True, help="E37 bench JSONL (eval only, not shipped)")
    ap.add_argument("--data_dir", default=None, help="default $HOROPTER_DATA/data/v3/e38")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    global D, BENCH, RESULTS
    D, BENCH, RESULTS = Path(a.data_dir) if a.data_dir else D, a.bench, Path(a.results)
    build() if a.cmd == "build" else run(a.node_model, a.tag, a.oneshot_model)


if __name__ == "__main__":
    main()
