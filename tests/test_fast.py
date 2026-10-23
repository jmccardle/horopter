# SPDX-License-Identifier: MIT
"""The fast path (horopter.fast) against the release path, through the served endpoint.

Needs an exported model directory, else every test is skipped:
    HOROPTER_TEST_MODEL=path/to/horopter-0.4B python -m pytest -s tests/test_fast.py

cpu   (always, if the model is set) fp32 state cache, eager: same answers as the release path, max |dp| < 1e-4,
      identical evidence spans, over more states than cache slots (eviction) asked in rotating order, twice.
cuda  (if a GPU is present) graphs + compile + Flex + multi-adapter pass with bf16 weights, against the release path
      (fp32 weights, bf16 autocast): max |dp| <= 0.06 and an answer may differ only where the release path's top two
      options are within 2 * max |dp| (a tie). Replay check: 8 states (40 to 1,900 tokens) against 3 graph slots, warm,
      after a cache reset, and warm again, each request's answers equal to its first answer within 1e-3: a captured
      graph must never read a freed or overwritten buffer.
"""
import copy
import os

import pytest

MODEL = os.environ.get("HOROPTER_TEST_MODEL")
pytestmark = pytest.mark.skipif(not (MODEL and os.path.exists(os.path.join(MODEL or "", "config.json"))),
                                reason="set HOROPTER_TEST_MODEL to an exported model directory")

REFUND = ("Refunds are available within 30 days of purchase for unopened items. Opened items may be returned for store "
          "credit only. The customer bought headphones 12 days ago and returned them unopened with the receipt. ")
LEASE = ("The tenant must give 60 days' written notice before moving out. Smoking is prohibited anywhere on the "
         "property. The tenant gave notice by email 45 days before moving out. ")
FILL = ("Section {i}. Employees may carry over up to five unused leave days into the next calendar year, provided the "
        "request is filed before December 15 and approved by a line manager. ")
QS = {"refund": {"type": "noul", "instructions": "Is the customer entitled to a refund?"},
      "notice": {"type": "noul", "instructions": "Did the tenant give enough notice?"},
      "smoke": {"type": "choice", "instructions": "May the tenant smoke on the balcony?",
                "criteria": {"yes": "smoking there is allowed", "no": "smoking there is prohibited",
                             "unclear": "the text does not say"}},
      "strict": {"type": "score", "instructions": "How strict is this policy?", "criteria": ["lenient", "moderate", "strict"]}}


def state_of(tok, n, tag=""):
    s, i = tag + REFUND + LEASE, 1
    while len(tok(s, add_special_tokens=False)["input_ids"]) < n:
        s += FILL.format(i=i)
        i += 1
    return tok.decode(tok(s, add_special_tokens=False)["input_ids"][:n])


def requests(tok, models):
    out = []
    for n in (40, 150, 500):
        s = state_of(tok, n)
        for m in models:
            out.append({"model": m, "state": s, "questions": QS})
        out.append({"model": models[0], "state": s, "questions": {"refund": QS["refund"]},
                    "mbnli": {"evidence": {"mode": "chosen"}}})
        out.append({"model": models[-1], "state": s, "questions": QS, "mbnli": {"evidence": {"mode": "all"}}})
    out.append({"model": models[0], "state": {"order": {"days_since_purchase": 12, "opened": False}},
                "questions": {"refund": QS["refund"]}})
    return out


def endpoint(st, fast=None):
    from fastapi.testclient import TestClient

    from horopter.server import build_app
    c = TestClient(build_app(st, prefix="horopter", aliases=(), fast=fast))

    def post(body):
        r = c.post("/v1/systemone", json=body)
        assert r.status_code == 200, r.text
        return r.json()
    return post


def probs(body):
    return {q: [float(x) for x in (a.get("probabilities") or {"t": a["noul"], "f": 1 - a["noul"]}).values()]
            for q, a in body["answers"].items()}


def spans(body):
    return {q: sorted((e["start"], e["end"], e["role"], e["option"]) for e in a["horopter"].get("evidence", []))
            for q, a in body["answers"].items()}


def models_of(st):
    ms = ["horopter-" + a for a in sorted(st.bank) if a not in ("router", "evidence") and not a.startswith("ev_")]
    return (["horopter"] if "router" in st.bank else []) + ms[:3]


def compare(ref, got, tol):
    dp, flips = 0.0, []
    for q, r in probs(ref).items():
        g = probs(got)[q]
        d = max(abs(a - b) for a, b in zip(r, g))
        dp = max(dp, d)
        if max(range(len(r)), key=r.__getitem__) != max(range(len(g)), key=g.__getitem__):
            rs = sorted(r, reverse=True)
            flips.append(rs[0] - rs[1])
    return dp, flips


def test_fast_cpu():
    import torch

    from horopter import fast, runtime
    torch.manual_seed(0)
    ref_st = runtime.load(MODEL, device="cpu")
    st = runtime.load(MODEL, device="cpu")
    fp = fast.FastPath(st, compile=False, cache_size=2)
    ref, got = endpoint(ref_st), endpoint(st, fp)
    reqs = requests(st.tok, models_of(st))
    worst = 0.0
    for rnd in range(2):                        # 4 states vs 2 cache slots, rotated: misses, evictions and hits
        for b in reqs[rnd:] + reqs[:rnd]:
            r, g = ref(copy.deepcopy(b)), got(copy.deepcopy(b))
            dp, flips = compare(r, g, 1e-4)
            worst = max(worst, dp)
            assert dp < 1e-4 and not flips, (b["model"], dp)
            assert spans(r) == spans(g)
    assert fp.split.hits > 0 and fp.split.misses > 0, "the state cache was not used"
    # unrounded: the answer dicts themselves (the wire format rounds to 4 decimals)
    raw = 0.0
    for n in (40, 500):
        s = state_of(st.tok, n)
        for a in sorted(st.bank)[:4]:
            ref_st.use(a)
            r = ref_st.answer(s, QS, evidence=True)
            g = fp.call(lambda: fp.answer(s, QS, {q: {"adapter": a} for q in QS}, True))
            for q in QS:
                raw = max(raw, float(abs(r[q]["p"] - g[q]["p"]).max()), abs(r[q]["unrel"] - g[q]["unrel"]))
                if "evidence" not in st.bank:
                    raw = max(raw, max(float(abs(x[1] - y[1]).max()) for x, y in zip(r[q]["ev"], g[q]["ev"])))
    assert raw < 1e-4, raw
    fp.close()
    print(f"cpu: {2 * len(reqs)} requests, max |dp| {worst:.2e} (served, rounded), {raw:.2e} (raw); "
          f"cache hits {fp.split.hits} misses {fp.split.misses}")


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
def test_fast_cuda_and_replay():
    from horopter import fast, runtime
    ref_st = runtime.load(MODEL, device="cuda")
    st = runtime.load(MODEL, device="cuda")
    fp = fast.FastPath(st, slots=3)
    ref, got = endpoint(ref_st), endpoint(st, fp)
    reqs = requests(st.tok, models_of(st))
    tol, worst = 0.06, 0.0
    for b in reqs:
        r, g = ref(copy.deepcopy(b)), got(copy.deepcopy(b))
        dp, flips = compare(r, g, tol)
        worst = max(worst, dp)
        assert dp <= tol, (b["model"], dp)
        assert all(gap <= 2 * tol for gap in flips), (b["model"], flips)
    # replay / eviction: more states than slots, rotating order, warm -> reset -> warm
    states = [state_of(st.tok, n, f"[{i}] ") for i, n in enumerate((40, 150, 200, 330, 500, 700, 1000, 1900))]
    ms = models_of(st)
    first, drift = {}, 0.0
    for phase in range(3):
        if phase == 1:
            fp.call(fp.split.reset)
        for rnd in range(2):
            for i in list(range(len(states)))[rnd:] + list(range(len(states)))[:rnd]:
                for m in ms:
                    p = probs(got({"model": m, "state": states[i], "questions": QS}))
                    for q, v in p.items():
                        f = first.setdefault((i, m, q), v)
                        drift = max(drift, max(abs(a - b) for a, b in zip(v, f)))
    fp.close()
    print(f"cuda: {len(reqs)} requests vs release max |dp| {worst:.4f}; replay drift {drift:.2e}; multi-adapter "
          f"passes {fp.ms.passes}, graphs {fp.ms.captures} + {fp.split.captures} state/question")
    assert drift <= 1e-3
    assert fp.ms.passes > 0 and fp.ms.captures > 0 and fp.split.captures > 0, "the graphed paths were not used"
