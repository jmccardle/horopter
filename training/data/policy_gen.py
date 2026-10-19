# SPDX-License-Identifier: MIT
"""E38b: programmatic policy generator v0 (ROADMAP W2.1), JF100-shaped.

A domain is a set of checks on a case (boolean or numeric threshold), each with the outcome it gives when the
case fails it, plus a default outcome when every check passes. One item = k checks in a stated precedence order
("check A first, then B ..."), optionally one exemption ("members with X skip the B check"), a case, and a
4-option choice question. The answer comes from executing the rules: the first non-exempt failed check in order
gives its outcome, else the default.

- depth = number of checks evaluated to reach the answer (+1 when an exemption had to be consulted).
- Two registers per item: "json" (state = {"policy": ..., "case": {...}}, as JF100) and "prose" (one text state,
  so evidence spans can be scored). gold_span = char span of the deciding rule sentence in the prose state.
- twin = the same rules with one case value changed so the answer changes (a JF100-style counterfactual pair).
- nodes = one noul per check, "does the case pass this check?", with gold: the interpreter's node questions.

    python training/data/policy_gen.py --n 600 --seed 0 > $HOROPTER_DATA/data/v3/e38/policy_v0.jsonl   # E38b eval set

Seeds in use: 0 = the policy_v0 eval set, 1 = check-adapter rows (node_data.py), 2 = evidence-adapter policy items
(train_evidence.py --n_policy), 3 = the beta-2 policy stream (build_prod2.py).
"""
import argparse
import json
import random
import sys

# domain -> (default outcome, question, [check]); check = (field, kind, pass phrase, fail outcome, value sampler)
#   kind "bool": passes when the field is True;  kind "max"/"min": passes when value <= / >= threshold t.
DOMAINS = {
    "retail_refund": ("refund_approved", "Which outcome applies to this refund request?", [
        ("receipt", "bool", "the customer has a receipt", "missing_receipt", None),
        ("age_days", "max", "the item was bought no more than {t} days ago", "too_old", (7, 60)),
        ("unused", "bool", "the item is unused", "used_item", None),
        ("amount_usd", "max", "the refund is at most ${t}", "needs_manager_approval", (50, 500)),
        ("original_payment", "bool", "the original payment card is presented", "store_credit_only", None),
    ]),
    "hr_leave": ("leave_approved", "What is the decision on this leave request?", [
        ("notice_days", "min", "the request is made at least {t} days in advance", "insufficient_notice", (3, 30)),
        ("tenure_months", "min", "the employee has at least {t} months of service", "not_yet_eligible", (3, 12)),
        ("balance_days", "min", "the employee has at least {t} days of leave balance", "unpaid_leave_only", (1, 10)),
        ("manager_signed", "bool", "the manager has signed the form", "pending_manager", None),
        ("blackout", "bool", "the dates are outside the blackout period", "denied_blackout", None),
    ]),
    "robot_sop": ("proceed", "What should the robot do?", [
        ("estop_clear", "bool", "the emergency stop is released", "halt", None),
        ("battery_pct", "min", "the battery is at least {t}%", "return_to_dock", (15, 40)),
        ("payload_kg", "max", "the payload is at most {t} kg", "request_assist", (5, 40)),
        ("zone_clear", "bool", "the work zone is clear of people", "wait_for_clearance", None),
        ("lidar_ok", "bool", "the lidar self-test passed", "run_diagnostics", None),
    ]),
    "insurance_claim": ("pay_claim", "How should the claim be handled?", [
        ("policy_active", "bool", "the policy was active on the loss date", "deny_lapsed", None),
        ("reported_days", "max", "the loss was reported within {t} days", "deny_late_report", (14, 90)),
        ("photos", "bool", "photos of the damage were submitted", "request_evidence", None),
        ("estimate_usd", "max", "the estimate is at most ${t}", "send_adjuster", (1000, 20000)),
        ("not_flood", "bool", "the damage was not caused by flooding", "deny_flood_exclusion", None),
    ]),
}
EXEMPT = [("member_gold", "gold members"), ("vip_ticket", "customers with a VIP ticket"),
          ("safety_officer", "cases signed off by a safety officer")]


def sample_case(rng, checks, tvals, ok):
    """ok[i]: whether the case passes check i. Numeric values sit a random distance from the threshold, never on it."""
    case = {}
    for (f, kind, _, _, rngv), t, o in zip(checks, tvals, ok):
        if kind == "bool":
            case[f] = o
        else:
            d = rng.randint(1, max(2, (rngv[1] - rngv[0]) // 3))
            up = (kind == "min") == o          # min-check passes above t, max-check passes below t
            case[f] = t + d if up else max(0, t - d)
            if passes((f, kind), t, case[f]) != o:   # clipped at 0: fall back to a passing value
                case[f] = t + d
    return case


def passes(check, t, v):
    kind = check[1]
    return bool(v) if kind == "bool" else (v <= t if kind == "max" else v >= t)


def execute(checks, tvals, case, exempt):
    """-> (outcome, index of deciding check or None, depth)"""
    depth = 0
    for i, (c, t) in enumerate(zip(checks, tvals)):
        depth += 1
        if not passes(c, t, case[c[0]]):
            if exempt and exempt[1] == i:
                depth += 1
                if case[exempt[0]]:
                    continue
            return c[3], i, depth
    return None, None, depth


def phrase(c, t):
    return c[2].format(t=t)


def render(dom, checks, tvals, exempt, case, default):
    names = [c[0].replace("_", " ") for c in checks]
    sents = [f"A request is approved only if {' and '.join(phrase(c, t) for c, t in zip(checks, tvals))}."
             if len(checks) > 1 else f"A request is approved only if {phrase(checks[0], tvals[0])}."]
    sents.append("Check the conditions in this order: " + ", then ".join(names) + ". The first failed check decides.")
    rule_spans = []
    body = " ".join(sents)
    for c, t in zip(checks, tvals):
        s = f" Failing the {c[0].replace('_', ' ')} check ({phrase(c, t)}) gives the outcome {c[3]}."
        rule_spans.append((len(body) + 1, len(body) + len(s)))
        body += s
    if exempt:
        f, who = exempt[0], dict(EXEMPT)[exempt[0]]
        s = f" Exception: {who} skip the {names[exempt[1]]} check."
        exempt_span = (len(body) + 1, len(body) + len(s))
        body += s
    else:
        exempt_span = None
    body += f" If every check passes, the outcome is {default}."
    policy = body.strip()
    shift = len(body) - len(policy)
    rule_spans = [(a - shift, b - shift) for a, b in rule_spans]
    exempt_span = (exempt_span[0] - shift, exempt_span[1] - shift) if exempt_span else None
    case_txt = "Case: " + "; ".join(f"{k.replace('_', ' ')} = {json.dumps(v)}" for k, v in case.items()) + "."
    return policy, policy + "\n" + case_txt, rule_spans, exempt_span


def make(rng, idx):
    dom = rng.choice(sorted(DOMAINS))
    default, question, pool = DOMAINS[dom]
    k = rng.randint(1, 5)
    checks = rng.sample(pool, k)
    tvals = [None if c[1] == "bool" else rng.randint(*c[4]) for c in checks]
    exempt = None
    if k >= 2 and rng.random() < 0.4:
        exempt = (rng.choice(EXEMPT)[0], rng.randrange(k))
    target = rng.randint(0, k)          # deciding check (k = all pass -> default): balances depth
    ok = [True] * target + [False] + [rng.random() < 0.5 for _ in range(k - target - 1)] if target < k else [True] * k
    case = sample_case(rng, checks, tvals, ok)
    if exempt:
        case[exempt[0]] = rng.random() < 0.5
    out, di, depth = execute(checks, tvals, case, exempt)
    out = out or default
    # twin: change one case value so the outcome changes
    twin = None
    for f in rng.sample(list(case), len(case)):
        c2 = dict(case)
        c2[f] = (not c2[f]) if isinstance(c2[f], bool) else next(
            (v for v in (tvals[[c[0] for c in checks].index(f)] + d for d in (-3, 3, -10, 10)) if v >= 0), c2[f]) \
            if f in [c[0] for c in checks] else c2[f]
        o2 = execute(checks, tvals, c2, exempt)[0] or default
        if o2 != out:
            twin = c2
            break
    outcomes = sorted({c[3] for c in pool} | {default})
    distract = [o for o in outcomes if o != out]
    rng.shuffle(distract)
    opts = [out] + distract[:3]
    rng.shuffle(opts)
    letters = "ABCD"

    def item(cs, tag):
        o, d_i, dep = execute(checks, tvals, cs, exempt)
        o = o or default
        pol, prose, rspans, espan = render(dom, checks, tvals, exempt, cs, default)
        q = {"q": {"type": "choice", "instructions": question,
                   "criteria": {letters[i]: v for i, v in enumerate(opts)}}}
        gold = letters[opts.index(o)] if o in opts else None
        nodes = [{"instructions": f"Does the case satisfy this condition: {phrase(c, t)}?",
                  "gold": passes(c, t, cs[c[0]]), "field": c[0]} for c, t in zip(checks, tvals)]
        span = rspans[d_i] if d_i is not None else None
        return {"id": f"pol{idx:05d}{tag}", "pair": f"pol{idx:05d}", "domain": dom, "k": k, "depth": dep,
                "exempt": exempt is not None,
                "exempt_field": exempt[0] if exempt else None, "exempt_who": dict(EXEMPT)[exempt[0]] if exempt else None,
                "exempt_check": checks[exempt[1]][0] if exempt else None,
                "fail_outcome": {c[0]: c[3] for c in checks}, "order": [c[0] for c in checks], "default": default,
                "decided_by": "default" if d_i is None else checks[d_i][0],
                "json_state": {"policy": pol, "case": cs}, "prose_state": prose, "questions": q,
                "gold": {"q": gold}, "gold_span": span, "nodes": nodes}

    items = [item(case, "a")]
    if twin is not None:
        items.append(item(twin, "b"))
    return [it for it in items if it["gold"]["q"] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for i in range(a.n):
        for it in make(rng, i):
            sys.stdout.write(json.dumps(it) + "\n")


if __name__ == "__main__":
    main()
