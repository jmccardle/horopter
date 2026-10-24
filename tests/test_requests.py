# SPDX-License-Identifier: MIT
"""Request checks, the evidence shorthand, /v1/tokenize and passes of max_batch questions (0.0.5).

The schema tests run without weights; the rest need a model (CPU):

    HOROPTER_TEST_MODEL=/path/to/horopter-0.4B python -m pytest -s tests/test_requests.py
"""
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
pytest.importorskip("pydantic")
from pydantic import ValidationError  # noqa: E402

from horopter.api import DecideRequest  # noqa: E402
from horopter.service import _chunks  # noqa: E402

MODEL = os.environ.get("HOROPTER_TEST_MODEL")
STATE = ("The agent ran pytest and 3 tests failed. Then it edited src/a.py, reran pytest, and all tests passed. "
         "It did not touch the docs.")
Q = {"q": {"type": "noul", "instructions": "Did all tests pass at the end?"}}


@pytest.mark.parametrize("body, loc", [
    ({"state": "x", "evidnce": "chosen"}, ("evidnce",)),
    ({"state": "x", "questions": {"q": {"type": "noul", "question": "a"}}}, ("questions", "q", "noul", "question")),
    ({"state": "x", "horopter": {"debgu": True}}, ("horopter", "debgu")),
    ({"state": "x", "options": {"return_text": True}}, ("options", "return_text")),
    ({"state": "x", "fields": {"f": {"type": "text", "instruction": "a"}}}, ("fields", "f", "instruction")),
])
def test_unknown_keys_are_refused(body, loc):
    with pytest.raises(ValidationError) as e:
        DecideRequest.model_validate(body)
    assert [(x["type"], x["loc"]) for x in e.value.errors()] == [("extra_forbidden", loc)]


def test_evidence_shorthand_parses():
    assert DecideRequest.model_validate({"state": "x", "evidence": "chosen"}).evidence == "chosen"
    assert DecideRequest.model_validate({"state": "x", "evidence": {"mode": "all"}}).evidence == {"mode": "all"}
    with pytest.raises(ValidationError):
        DecideRequest.model_validate({"state": "x", "evidence": "some"})


def test_chunks():
    assert _chunks("abcde", 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert _chunks("abc", 0) == [["a", "b", "c"]]


@pytest.fixture(scope="module")
def svc():
    if not MODEL:
        pytest.skip("set HOROPTER_TEST_MODEL to a model directory")
    from horopter import runtime
    from horopter.service import Service
    return Service(runtime.load(MODEL, device="cpu", adapters=False), max_batch=3)


def _err(svc, body):
    from horopter.service import ServiceError
    with pytest.raises(ServiceError) as e:
        svc.decide(body)
    return e.value


def test_service_refuses_unknown_keys(svc):
    e = _err(svc, {"state": STATE, "questions": Q, "evidnce": "chosen"})
    assert e.status == 400 and e.detail[0]["loc"] == ["evidnce"]
    e = _err(svc, {"state": STATE, "questions": Q, "horopter": {"evidence": {"mode": "chosen", "thr": 0.3}}})
    assert e.status == 400 and "thr" in e.detail


def test_top_level_evidence_is_the_nested_option(svc):
    a = svc.decide({"state": STATE, "questions": Q, "evidence": "chosen"})
    b = svc.decide({"state": STATE, "questions": Q, "horopter": {"evidence": {"mode": "chosen"}}})
    assert a["answers"] == b["answers"] and "evidence" in a["answers"]["q"]["horopter"]
    e = _err(svc, {"state": STATE, "questions": Q, "evidence": "all", "horopter": {"evidence": {"mode": "chosen"}}})
    assert e.status == 400


def test_passes_give_the_same_answers(svc):
    qs = {f"q{i}": {"type": "noul", "instructions": t} for i, t in enumerate([
        "Did all tests pass at the end?", "Did any test fail?", "Was src/a.py edited?", "Were the docs changed?",
        "Did the agent run pytest twice?", "Did the agent delete a file?", "Was the run successful?"])}
    qs["kind"] = {"type": "choice", "instructions": "What did the agent mostly do?",
                  "criteria": {"test": "ran tests", "docs": "wrote docs", "deploy": "deployed"}}
    fields = {"file": {"type": "text", "instructions": "The edited file"}}
    body = {"state": STATE, "questions": qs, "fields": fields, "evidence": "chosen"}
    many = svc.decide(body)                       # max_batch=3: 9 questions and fields in 3 passes
    svc.max_batch = 0
    try:
        one = svc.decide(body)
    finally:
        svc.max_batch = 3
    for k, a in one["answers"].items():
        b = many["answers"][k]
        for f in ("noul", "choice", "score", "value", "p_null", "span"):
            if isinstance(a.get(f), float):
                assert b[f] == pytest.approx(a[f], abs=2e-4), (k, f)
            else:
                assert b.get(f) == a.get(f), (k, f)
        if "horopter" in a:
            assert [s["text"] for s in b["horopter"]["evidence"]] == [s["text"] for s in a["horopter"]["evidence"]]


def test_tokenize_matches_decide(svc):
    t = svc.tokenize({"state": STATE})
    r = svc.decide({"state": STATE, "questions": Q, "evidence": "chosen", "options": {"return_state_text": True}})
    assert t["state_text"] == r["state_text"] == STATE
    assert t["state_tokens"] == r["usage"]["state_tokens"] and not t["truncated"]
    assert t["state_id"] == r["horopter"]["state_id"]
    assert len(t["offsets"]) == t["state_tokens"] and t["read_chars"] == len(STATE)
    assert all(0 <= a <= b <= len(STATE) for a, b in t["offsets"])
    for s in r["answers"]["q"]["horopter"]["evidence"]:  # span offsets index state_text
        assert STATE[s["start"]:s["end"]] == s["text"]


def test_tokenize_json_and_limit(svc):
    t = svc.tokenize({"state": {"b": [1, 2], "a": "x"}})
    assert t["state_text"] == '{"a":"x","b":[1,2]}'
    long = " ".join(["word"] * (svc.st.col.ms + 50))
    t = svc.tokenize({"state": long})
    assert t["truncated"] and t["limit"] == svc.st.col.ms - 2 and t["state_tokens"] > t["limit"]
    assert t["read_chars"] == t["offsets"][t["limit"] - 1][1]
    r = svc.decide({"state": long, "questions": Q, "options": {"allow_truncation": True, "return_state_text": True}})
    assert len(r["state_text"]) == t["read_chars"] and r["usage"]["state_tokens"] == t["limit"]
    assert _err(svc, {"state": long, "questions": Q}).status == 413
    from horopter.service import ServiceError
    with pytest.raises(ServiceError) as e:
        svc.tokenize({"state": STATE, "model": "nope"})
    assert e.value.status == 404
    assert svc.health()["max_state_tokens"] == t["limit"]
