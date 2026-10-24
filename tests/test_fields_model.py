# SPDX-License-Identifier: MIT
"""Field extraction end to end on a model with the extract adapter loaded (CPU):

    HOROPTER_TEST_MODEL=/path/to/horopter-0.4B HOROPTER_TEST_EXTRACT=/path/to/extract python -m pytest -s tests/test_fields_model.py

HOROPTER_TEST_EXTRACT defaults to the model's bundled adapters/extract.
"""
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
MODEL = os.environ.get("HOROPTER_TEST_MODEL")

STATE = ("Hi team, the draft is attached as quarterly.report.final.pdf. Questions to support@homecomfi.toronto.ca "
         "please, or call 0161 496 0000. The total due is $1,240.50 by 14 November 2026.")
FIELDS = {"file": {"type": "text", "instructions": "The attached document"},
          "email": {"type": "text", "instructions": "The support email address"},
          "phone": {"type": "text", "instructions": "The phone number"},
          "total": {"type": "number", "instructions": "The total due"},
          "due": {"type": "date", "instructions": "The due date"},
          "iban": {"type": "text", "instructions": "The IBAN to pay into"}}


@pytest.fixture(scope="module")
def h():
    if not MODEL:
        pytest.skip("set HOROPTER_TEST_MODEL to a model directory")
    ex = os.environ.get("HOROPTER_TEST_EXTRACT") or os.path.join(MODEL, "adapters", "extract")
    if not os.path.isdir(ex):
        pytest.skip(f"no extract adapter at {ex} (set HOROPTER_TEST_EXTRACT)")
    import horopter
    return horopter.local(MODEL, device="cpu", adapters_from={"extract": ex})


@pytest.fixture(scope="module")
def r(h):
    return h.decide(STATE, fields=FIELDS, evidence="chosen")


def test_fields_use_the_extract_adapter(r):
    assert r["horopter"]["adapters_used"] == ["extract"]


def test_values(r):
    a = r["answers"]
    assert a["email"]["value"] == "support@homecomfi.toronto.ca"
    assert a["phone"]["value"] == "0161 496 0000"
    assert a["total"]["value"] == 1240.5
    assert a["due"]["value"] == "2026-11-14"


@pytest.mark.xfail(reason="known beta limit (docs/API.md): free-text values lose inner punctuation; a training fix",
                   strict=False)
def test_free_text_value_kept_whole_across_dots(r):
    # no type pattern: the value is the support span itself
    assert r["answers"]["file"]["value"] == "quarterly.report.final.pdf"


def test_absent_value_is_null(r):
    a = r["answers"]["iban"]
    assert a["value"] is None and a["p_null"] > 0.5


def test_spans_point_at_the_values(r):
    for k in ("email", "phone"):
        a, b = r["answers"][k]["span"]
        assert STATE[a:b] == r["answers"][k]["value"]
    assert STATE[slice(*r["answers"]["total"]["span"])] == "$1,240.50"
    assert STATE[slice(*r["answers"]["due"]["span"])] == "14 November 2026"
