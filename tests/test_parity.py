# SPDX-License-Identifier: MIT
"""Parity: the release runtime / server against the research code it was ported from, on CPU in fp32.

Needs two things, else every test is skipped:
    HOROPTER_RESEARCH_ROOT   a checkout of the research repository (its experiments/v3/e24_dryrun_router/serve.py
                             and outputs/v3/e12/<run>/model.pt), importable in this Python environment
    HOROPTER_PARITY_EXPORT   the same run exported by scripts/export_hf.py, e.g.
        python scripts/export_hf.py --ckpt $HOROPTER_RESEARCH_ROOT/outputs/v3/e12/prod_e400m/model.pt \
            --base jhu-clsp/ettin-encoder-400m --max_state 2048 \
            --adapter $HOROPTER_RESEARCH_ROOT/outputs/v3/e24/prod400m/check \
            --abstain_head $HOROPTER_RESEARCH_ROOT/outputs/v3/e35/abstain_lr_T2.json --out $HOROPTER_PARITY_EXPORT
Optional: HOROPTER_PARITY_TRUNK (research trunk name, default prod400m), HOROPTER_PARITY_REPORT (write the max
differences as JSON there).

    CUDA_VISIBLE_DEVICES="" python -m pytest -s tests/test_parity.py
Checks per request (about 20: noul / choice / score, text and JSON states, several questions per state, evidence on):
probabilities, stance rows, p(unrelated) and the 21 abstention features (max abs diff < 1e-4), raw evidence channel
scores (< 1e-4), and the served JSON (evidence spans identical, abstain.p and every rounded number equal up to one
unit in the last place).
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
RESEARCH = os.environ.get("HOROPTER_RESEARCH_ROOT")
EXPORT = os.environ.get("HOROPTER_PARITY_EXPORT")
TRUNK = os.environ.get("HOROPTER_PARITY_TRUNK", "prod400m")
TOL = 1e-4

pytestmark = pytest.mark.skipif(
    not (RESEARCH and EXPORT and Path(RESEARCH, "experiments/v3/e24_dryrun_router/serve.py").exists()
         and Path(EXPORT, "config.json").exists()),
    reason="set HOROPTER_RESEARCH_ROOT and HOROPTER_PARITY_EXPORT (see the module docstring)")

REFUND = ("Refunds are issued within 30 days of purchase if the item is unused and in its original packaging. "
          "Items bought on clearance cannot be refunded. The customer bought a lamp on clearance 12 days ago; "
          "it is unused and still boxed.")
LEASE = ("The tenant must give 60 days' written notice before ending the lease. Pets are allowed with a $300 deposit. "
         "Smoking is prohibited anywhere on the property. The tenant gave notice by email 45 days before moving out "
         "and kept a cat, for which a deposit of $300 was paid.")
TRIAL = ("Patients aged 18 to 65 with type 2 diabetes and an HbA1c between 7.0% and 10.0% are eligible. Patients on "
         "insulin are excluded. Candidate: 52-year-old woman, type 2 diabetes, HbA1c 8.2%, treated with metformin only.")
REVIEW = "The battery lasts two days, but the screen scratches easily and support never answered my emails."
UNREL = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
ORDER = {"order": {"id": "A-1093", "total": 1250.0, "currency": "EUR", "items": 3, "express": True},
         "customer": {"tier": "gold", "country": "DE", "orders_last_year": 14},
         "policy": "Express shipping is free for gold customers on orders over 1000 EUR."}
TICKET = {"ticket": {"subject": "Cannot log in", "body": "Password reset email never arrives, tried 3 times.",
                     "priority": "unset", "product": "web app"}}

YESNO = {"type": "noul", "instructions": "Is the customer entitled to a refund?"}
YESNO_CRIT = {"type": "noul", "instructions": "Did the tenant give enough notice?",
              "criteria": {"true": "the notice period was met", "false": "the notice was too short"}}
NLI_CHOICE = {"type": "choice", "instructions": "Is the candidate eligible for the trial?",
              "criteria": {"eligible": "meets every inclusion criterion and no exclusion",
                           "ineligible": "fails a criterion", "unclear": "the text does not settle it"}}
SENT = {"type": "choice", "instructions": "What is the overall sentiment of the review?",
        "criteria": {"positive": "", "mixed": "both praise and complaints", "negative": ""}}
SCORE = {"type": "score", "instructions": "How satisfied is the reviewer?",
         "criteria": ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"]}
SCORE3 = {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]}
PRIO = {"type": "choice", "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "payments and invoices", "auth": "accounts, passwords and login",
                     "shipping": "deliveries", "other": "anything else"}}


def requests():
    R = []
    for model in ("base", "check"):
        R += [
            (model, REFUND, {"q1": YESNO, "q2": {"type": "noul", "instructions": "Was the item bought on clearance?"}},
             "all"),
            (model, LEASE, {"notice": YESNO_CRIT, "pets": {"type": "noul", "instructions": "Are pets allowed?"},
                            "smoke": {"type": "noul", "instructions": "Did the tenant smoke on the property?"}}, "chosen"),
            (model, TRIAL, {"elig": NLI_CHOICE, "ins": {"type": "noul", "instructions": "Is the candidate on insulin?"}},
             "all"),
            (model, REVIEW, {"sent": SENT, "sat": SCORE}, "all"),
            (model, UNREL, {"r": YESNO, "s": SCORE3}, "chosen"),
            (model, ORDER, {"free": {"type": "noul", "instructions": "Is express shipping free for this order?"},
                            "tier": {"type": "choice", "instructions": "Customer tier?",
                                     "criteria": {"bronze": "", "silver": "", "gold": ""}}}, "all"),
            (model, TICKET, {"team": PRIO, "urg": SCORE3}, "off"),
            (model, REFUND + " " + LEASE, {"q1": YESNO, "notice": YESNO_CRIT, "sent": SENT}, "all"),
            (model, TRIAL, {"elig": NLI_CHOICE, "sat": SCORE, "x": {"type": "noul", "instructions": {"ask": "HbA1c < 9?"}}},
             "chosen"),
            (model, "Short.", {"q": {"type": "noul", "instructions": "Is this text in English?"}}, "all"),
        ]
    return R


@pytest.fixture(scope="module")
def pair():
    """(research app, research Student, release app, release Student)."""
    import torch
    torch.set_num_threads(int(os.environ.get("HOROPTER_PARITY_THREADS", "8")))
    sys.path.insert(0, str(Path(RESEARCH) / "experiments/v3/e24_dryrun_router"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.dont_write_bytecode = True   # never write caches into the research checkout
    import serve as rs   # research server (imports student / train_joint, DEV = cpu with no visible GPU)

    from horopter import runtime
    from horopter import server as hs
    cfg = json.load(open(Path(EXPORT) / "config.json"))
    tmp = Path(tempfile.mkdtemp(prefix="horopter_parity_"))
    for name in cfg["adapters"]:   # the research server reads every <name>/adapter.safetensors of one dir
        (tmp / name).mkdir()
        for f in ("adapter.safetensors", "adapter.json"):
            os.symlink(Path(EXPORT) / "adapters" / name / f, tmp / name / f)
    head = str(Path(EXPORT) / "abstain_head.json") if cfg.get("abstain_head") else ""
    r_app = rs.build_app(TRUNK, tmp, "cpu", cfg["max_state"], "mbnli-3-beta", head)
    r_st = next(c.cell_contents for r in r_app.routes if getattr(r, "path", "") == "/v1/systemone"
                for c in (r.endpoint.__closure__ or ()) if type(c.cell_contents).__name__ == "Student")
    h_st = runtime.load(EXPORT, device="cpu")
    h_app = hs.build_app(h_st, "horopter", ["mbnli-3-beta"])
    yield r_app, r_st, h_app, h_st
    shutil.rmtree(tmp, ignore_errors=True)


def _num_diff(a, b, path=""):
    """Max abs difference of numbers in two JSON trees (structure and strings must match)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {sorted(a)} vs {sorted(b)}"
        return max([_num_diff(a[k], b[k], f"{path}.{k}") for k in a] or [0.0])
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: lengths {len(a)} vs {len(b)}"
        return max([_num_diff(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))] or [0.0])
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        assert a == b, f"{path}: {a!r} vs {b!r}"
        return 0.0
    return abs(float(a) - float(b))


def _spans(body):
    return {q: a["mbnli"].get("evidence") for q, a in body["answers"].items() if "mbnli" in a}


def test_parity(pair):
    from fastapi.testclient import TestClient
    r_app, r_st, h_app, h_st = pair
    rc, hc = TestClient(r_app), TestClient(h_app)
    worst = {"p": 0.0, "rel": 0.0, "unrel": 0.0, "f": 0.0, "ev": 0.0, "json": 0.0, "abstain_p": 0.0}
    n_q = n_spans = 0
    by_model = {}
    for model, state, qs, ev in requests():
        # raw runtime outputs, adapter by adapter
        state_s = state if isinstance(state, str) else json.dumps(state, sort_keys=True, separators=(",", ":"))
        r_st.load_adapter(r_st.base_state if model == "base" else
                          {k: v for k, v in _bank(r_app)[model].items()}, model)
        h_st.load_adapter(h_st.bank[model], model)
        nq = {q: _norm(v, q) for q, v in qs.items()}
        ro = r_st.answer(state_s, nq, evidence=True)
        ho = h_st.answer(state_s, nq, evidence=True)
        by_model[(model, json.dumps(qs, sort_keys=True), state_s)] = ho
        for q in nq:
            for k in ("p", "rel", "f"):
                worst[k] = max(worst[k], float(np.abs(np.asarray(ro[q][k]) - np.asarray(ho[q][k])).max()))
            worst["unrel"] = max(worst["unrel"], abs(ro[q]["unrel"] - ho[q]["unrel"]))
            for (o1, e1), (o2, e2) in zip(ro[q]["ev"], ho[q]["ev"]):
                assert o1 == o2
                worst["ev"] = max(worst["ev"], float(np.abs(e1 - e2).max()))
            n_q += 1
        r_st.load_adapter(r_st.base_state, "base")
        r_st.loaded = "base"
        h_st.load_adapter(h_st.base_state, "base")
        h_st.loaded = "base"
        # served JSON, same request through both servers (research name vs release name)
        body = {"state": state, "questions": qs, "mbnli": {"evidence": {"mode": ev}}}
        ra = rc.post("/v1/systemone", json=dict(body, model=f"mbnli-3-beta-{model}"))
        ha = hc.post("/v1/systemone", json=dict(body, model=f"horopter-{model}"))
        ha2 = hc.post("/v1/systemone", json=dict(body, model=f"mbnli-3-beta-{model}"))   # the alias still works
        assert ra.status_code == ha.status_code == ha2.status_code == 200, (ra.text, ha.text, ha2.text)
        ra, ha, ha2 = ra.json(), ha.json(), ha2.json()
        assert ha2["answers"] == ha["answers"]
        assert _spans(ra) == _spans(ha), (model, qs, _spans(ra), _spans(ha))
        n_spans += sum(len(v or []) for v in _spans(ha).values())
        for q in qs:
            worst["abstain_p"] = max(worst["abstain_p"], abs(ra["answers"][q]["mbnli"]["abstain"]["p"] -
                                                           ha["answers"][q]["mbnli"]["abstain"]["p"]))
        worst["json"] = max(worst["json"], _num_diff(ra["answers"], ha["answers"]))
        assert ra["usage"] == ha["usage"]
    # the adapter swap is real: the check adapter changes some answers relative to the trunk alone
    eff = max(float(np.abs(by_model[k][q]["p"] - by_model[("base",) + k[1:]][q]["p"]).max())
              for k in by_model if k[0] != "base" and ("base",) + k[1:] in by_model for q in by_model[k])
    report = {"requests": len(requests()), "questions": n_q, "served_spans": n_spans,
              "max_abs_diff": {k: float(v) for k, v in worst.items()}, "max_adapter_effect_p": eff}
    print("\nparity:", json.dumps(report), flush=True)
    if os.environ.get("HOROPTER_PARITY_REPORT"):
        json.dump(report, open(os.environ["HOROPTER_PARITY_REPORT"], "w"), indent=1)
    for k in ("p", "rel", "unrel", "f", "ev"):
        assert worst[k] < TOL, (k, worst[k])
    assert len(by_model) == 1 or eff > 1e-3, "adapters had no effect: the swap is not working"
    assert worst["json"] <= 1e-4 + 1e-9 and worst["abstain_p"] <= 1e-4 + 1e-9, worst


def _norm(q, qid):
    from horopter.server import normalise
    return normalise(q, qid)


def _bank(app):
    """The research server's adapter bank (closure of its /v1/systemone endpoint)."""
    for r in app.routes:
        if getattr(r, "path", "") == "/v1/systemone":
            for c in r.endpoint.__closure__ or ():
                v = c.cell_contents
                if isinstance(v, dict) and "base" in v and isinstance(v.get("base"), dict):
                    return v
    raise RuntimeError("research adapter bank not found")
