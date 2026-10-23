# SPDX-License-Identifier: MIT
"""Package smoke test: runs without model weights (CI), unlike test_parity / test_fast."""
import pytest

import horopter

pytest.importorskip("pydantic")
from horopter import server  # noqa: E402


def test_version():
    assert isinstance(horopter.__version__, str) and horopter.__version__.count(".") == 2


def test_question_checks():
    assert server.check_question("q", {"type": "noul", "instructions": "x"}) is None
    assert server.check_question("q", {"type": "choice", "criteria": {"a": "1"}}) is not None
    assert server.check_question("q", {"type": "score", "criteria": ["low", "high"]}) is None
    assert server.check_question("q", {"type": "box"}) is not None


def test_render_state():
    assert server.render_state("plain") == "plain"
    assert server.render_state({"b": 1, "a": 2}) == '{"a":2,"b":1}'
