# SPDX-License-Identifier: MIT
"""Adapter references and on-demand loading (horopter.runtime.adapter_path, horopter.routing.prepare), without a
model or the network: the hub download is replaced by a local directory."""
from pathlib import Path

import pytest

from horopter import families as F
from horopter import routing as RT
from horopter import runtime as R


def adapter_dir(root, rel, **meta):
    d = Path(root) / rel
    d.mkdir(parents=True)
    (d / "adapter.safetensors").write_bytes(b"")
    if meta:
        import json
        (d / "adapter.json").write_text(json.dumps(meta))
    return d


@pytest.mark.parametrize("s,repo,rev,sub", [
    ("user/repo", "user/repo", None, ""),
    ("user/repo:refunds", "user/repo", None, "refunds"),
    ("user/repo@b3:refunds", "user/repo", "b3", "refunds"),
    ("user/repo:adapters/nli", "user/repo", None, "adapters/nli"),
])
def test_ref_parse(s, repo, rev, sub):
    r = R.Ref.parse(s)
    assert (r.repo, r.revision, r.sub) == (repo, rev, sub) and str(r) == s
    assert str(r.child("x")) == repo + (f"@{rev}" if rev else "") + ":x"


def test_bundled_name_from_a_local_model(tmp_path):
    adapter_dir(tmp_path, "adapters/extract")
    assert R.adapter_path("extract", tmp_path) == tmp_path / "adapters/extract"
    with pytest.raises(FileNotFoundError, match="it has: extract"):
        R.adapter_path("nope", tmp_path)


def test_hub_downloads_only_the_named_adapter(tmp_path, monkeypatch):
    adapter_dir(tmp_path, "adapters/refunds")
    asked = []
    monkeypatch.setattr(R, "_download", lambda ref, patterns: asked.append((str(ref), patterns)) or tmp_path)
    # a bundled name on a hub model: the model's repo and revision
    p, origin = R.adapter_path("refunds", R.Ref.parse("jm/horopter-1B@b3"), with_origin=True)
    assert p == tmp_path / "adapters/refunds" and str(origin) == "jm/horopter-1B@b3:refunds"
    assert asked == [("jm/horopter-1B@b3:refunds", ["adapters/refunds/*", "refunds/*"])]
    # another repo laid out like a model repo
    assert R.adapter_path("someone/retail:refunds") == tmp_path / "adapters/refunds"


class FakeSt:
    """bank / meta / sources like a Student; ensure() records what it loads."""

    def __init__(self, metas):
        self.metas, self.bank, self.meta, self.sources, self.loaded = metas, {"base": {}}, {}, {}, []

    def ensure(self, ref, check_trunk=True):
        name = str(ref).rpartition(":")[2].rpartition("/")[2]
        if name not in self.bank:
            self.bank[name], self.meta[name] = {}, dict(self.metas.get(name, {}), name=name)
            self.sources[name] = R.Ref.parse(ref) if ":" in str(ref) else None
            self.loaded.append(str(ref))
        return name


def test_prepare_loads_only_what_is_named():
    st = FakeSt({})
    assert RT.prepare(st) == (None, None, None) and st.loaded == []        # no flags: nothing is downloaded
    st = FakeSt({})
    serve, router, routes = RT.prepare(st, router="router")              # the bundled router brings its families
    assert router == "router" and st.loaded == ["router", *F.FAMILIES]
    assert [r for r, _ in routes] == list(F.FAMILIES)
    st = FakeSt({})
    RT.prepare(st, serve="extract")
    assert st.loaded == ["extract"]


def test_a_router_brings_its_own_targets_from_its_repo():
    st = FakeSt({"router": {"targets": {"refunds": "a refund decision", "triage": "which team gets a ticket"}}})
    serve, router, routes = RT.prepare(st, router="someone/horopter-1B-retail@v1:router")
    assert st.loaded == ["someone/horopter-1B-retail@v1:router", "someone/horopter-1B-retail@v1:refunds",
                         "someone/horopter-1B-retail@v1:triage"]
    rt = RT.configure(st, serve=serve, router=router, routes=routes)
    assert rt.targets == {"refunds": "a refund decision", "triage": "which team gets a ticket"}
