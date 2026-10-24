# SPDX-License-Identifier: MIT
"""Routing configuration (horopter.routing) without a model, and served routing on a model when one is given:

    HOROPTER_TEST_MODEL=/path/to/horopter-0.4B python -m pytest -s tests/test_routing.py
"""
import os
from types import SimpleNamespace

import pytest

from horopter import families as F
from horopter import routing as RT

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
MODEL = os.environ.get("HOROPTER_TEST_MODEL")


class WordTok:
    """A stand-in tokenizer: one token per word."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids):
        return " ".join(self.words[: len(ids)])

    def tokenize_for(self, text):
        self.words = text.split()


def fake_st(names):
    return SimpleNamespace(bank={n: {} for n in ["base", *names]}, meta={n: {"name": n} for n in names})


def test_defaults():
    st = fake_st(["router", "evidence", *F.FAMILIES])
    assert RT.configure(st).serve == "base"                                # no flags: the trunk alone, even if loaded
    rt = RT.configure(st, router="router")                                 # the bundled router: the bundled families
    assert rt.router == "router" and rt.layout == "state" and list(rt.targets) == list(F.FAMILIES)
    rt = RT.configure(fake_st(["router", "nli", "typed"]), router="router")   # only loaded families are options
    assert list(rt.targets) == ["typed", "nli"]


def test_serve_and_custom_routes():
    st = fake_st(["router", "refunds", "triage", *F.FAMILIES])
    st.meta["triage"]["description"] = "which team gets a ticket"
    assert RT.configure(st, serve="refunds").info() == {"serve": "refunds"}
    rt = RT.configure(st, router="base", routes=[("refunds", "a refund decision"), ("triage", None), ("nli", None)])
    assert rt.layout == "question"
    assert rt.targets == {"refunds": "a refund decision", "triage": "which team gets a ticket",
                          "nli": F.FAMILIES["nli"][1]}
    st.meta["triage"]["routing_layout"] = "state"                          # a router trained on the state layout
    assert RT.configure(st, router="triage", routes=[("nli", None), ("typed", None)]).layout == "state"
    # the bundled router over a different set of targets keeps its training layout
    assert RT.configure(st, router="router", routes=[("refunds", None), ("nli", None)]).layout == "state"


@pytest.mark.parametrize("kw,msg", [
    (dict(serve="nli", router="base"), "cannot be combined"),
    (dict(router="base", routes=[("nli", None)]), "at least 2 targets"),
    (dict(router="base", routes=[("nli", None), ("nope", None)]), "not loaded"),
    (dict(serve="nope"), "not loaded"),
    (dict(router="base", routes=[("nli", None), ("nli", None)]), "twice"),
])
def test_errors(kw, msg):
    with pytest.raises(RT.RoutingError, match=msg):
        RT.configure(fake_st(["nli", "typed"]), **kw)


def test_question_layout_fits():
    """The routed question is embedded and cut so that every option block fits the question block."""
    rt = RT.Routing(router="base", targets={"a": "first thing " * 5, "b": "second"})
    tok = WordTok()
    q = {"type": "choice", "instructions": "word " * 200, "criteria": {"x": "", "y": ""}}
    full = f"{rt.instructions} Question (choice): {q['instructions']} Options: x | y"
    tok.tokenize_for(full)
    rq, cut = rt.question(q, tok, max_q=64)
    longest = len(" Answer: a: first thing first thing first thing first thing first thing".split())
    assert cut and len(rq["instructions"].split()) + longest <= 63
    assert rq["criteria"] == rt.targets and rq["instructions"].startswith(rt.instructions)
    short = {"type": "noul", "instructions": "Is it late?"}
    rq, cut = rt.question(short, tok, max_q=64)
    assert not cut and rq["instructions"] == f"{rt.instructions} Question (noul): Is it late? Options: true | false"


@pytest.fixture(scope="module")
def model():
    if not MODEL:
        pytest.skip("set HOROPTER_TEST_MODEL to a model directory with its bundled adapters")
    from horopter import runtime
    return runtime.load(MODEL, device="cpu")


REQ = {"state": "The refund policy allows returns within 30 days with a receipt. The customer bought the kettle 12 "
                "days ago and has the receipt.",
       "questions": {"ok": {"type": "noul", "instructions": "Can the customer return the kettle?"},
                     "topic": {"type": "choice", "instructions": "What is this text about?",
                               "criteria": {"retail": "shopping", "sports": "sports", "law": "courts"}}},
       "horopter": {"debug": True}}


def test_served_routing(model):
    from horopter.service import Service
    out = Service(model).decide(dict(REQ, model="horopter"))   # no routing given: the trunk alone
    assert all(a["horopter"]["route"]["adapter"] == "base" for a in out["answers"].values())
    svc = Service(model, routing=RT.configure(model, router="router"))   # bundled router, state layout
    out = svc.decide(dict(REQ, model="horopter"))
    assert all(a["horopter"]["route"]["adapter"] in F.FAMILIES for a in out["answers"].values())
    svc = Service(model, routing=RT.configure(model, serve="nli"))
    out = svc.decide(dict(REQ, model="horopter"))
    assert {a["horopter"]["route"]["adapter"] for a in out["answers"].values()} == {"nli"}
    assert svc.decide(dict(REQ, model="horopter-typed"))["horopter"]["adapters_used"] == ["typed"]
    rt = RT.configure(model, router="base", routes=[("typed", None), ("classification", None)])
    svc = Service(model, routing=rt)
    out = svc.decide(dict(REQ, model="horopter"))
    for a in out["answers"].values():
        r = a["horopter"]["route"]
        assert r["adapter"] in ("typed", "classification") and set(r["router"]) == {"typed", "classification"}
    assert "router base (question layout) over typed, classification" in str(svc.models())
    assert svc.health()["routing"]["router"] == "base"


def test_loads_only_named_adapters():
    if not MODEL:
        pytest.skip("set HOROPTER_TEST_MODEL to a model directory with its bundled adapters")
    from horopter import runtime
    st = runtime.load(MODEL, device="cpu", adapters=False)
    assert set(st.bank) == {"base"}
    st = runtime.load(MODEL, device="cpu", adapters=["evidence"])
    assert set(st.bank) == {"base", "evidence"}
    serve, router, routes = RT.prepare(st, router="router")       # the bundled router brings its six families
    assert set(st.bank) == {"base", "evidence", "router", *F.FAMILIES}
    assert RT.configure(st, serve=serve, router=router, routes=routes).targets.keys() == F.FAMILIES.keys()


def test_trunk_check(model, tmp_path):
    import json
    import shutil
    src = next(p for p in (__import__("pathlib").Path(MODEL) / "adapters").iterdir() if p.name == "nli")
    dst = tmp_path / "other"
    shutil.copytree(src, dst)
    meta = json.load(open(dst / "adapter.json"))
    json.dump(dict(meta, p="some_other_trunk", name="other"), open(dst / "adapter.json", "w"))
    with pytest.raises(ValueError, match="trained on trunk"):
        model.add_adapter_from(dst)
    assert model.add_adapter_from(dst, check_trunk=False) == "other" and "other" in model.bank
    assert model.add_adapter_from(src, name="nli2") == "nli2"
