# SPDX-License-Identifier: MIT
"""Horopter: an encoder-only typed-decision model (state encoded once, question-blind; typed answers, relation,
evidence spans and abstention per question).

    pip install horopter              the HTTP client only (horopter.connect): no model, no heavy dependencies
    pip install "horopter[client]"    + httpx (connection pooling)
    pip install "horopter[torch]"     run the model in this process (horopter.local, horopter.load)
    pip install "horopter[server]"    + the HTTP server (python -m horopter.server)
"""
from .client import Client, HoropterError, Local  # noqa: F401

__version__ = "0.0.4"

_TORCH = ("abstain", "evidence", "families", "fields", "losses", "model", "packing", "runtime", "quant", "fast",
          "service", "server", "api", "routing")
_NEED = 'this needs the torch backend: pip install "horopter[torch]" (or "horopter[server]" for the HTTP server)'


def connect(url="http://127.0.0.1:8420", model=None, **kw):
    """A client for a running server. -> horopter.Client (decide / models / health)."""
    return Client(url, model=model, **kw)


def local(path_or_hub_id="jmccardle/horopter-0.4B", device=None, fast=False, model=None, adapters=None,
          adapters_from=None, serve=None, router=None, routes=None, route_instructions=None, route_layout=None,
          **load_kw):
    """Load a model in this process and serve it through the same interface as connect(). -> horopter.Local
    Like the server: the trunk alone unless adapters are named, and only named adapters are downloaded.
    adapters: references to load (bundled names, directories, "user/repo[@rev][:NAME]"; True = every bundled one).
    adapters_from: {name: reference} to load under names of your choice (or a list of references).
    serve / router / routes / route_*: horopter.routing (routes: [(reference, description)]); whatever they name is
    loaded too, and a router without routes routes over the targets its adapter.json lists."""
    st = load(path_or_hub_id, device=device, adapters=adapters if adapters is not None else False, **load_kw)
    srcs = adapters_from.items() if isinstance(adapters_from, dict) else [(None, s) for s in adapters_from or ()]
    for name, src in srcs:
        st.add_adapter_from(src, name)
    from . import routing as _routing
    serve, router, routes = _routing.prepare(st, serve, router, routes)
    rt = _routing.configure(st, serve=serve, router=router, routes=routes, instructions=route_instructions,
                            layout=route_layout)
    fp = None
    if fast:
        from . import fast as _fast
        fp = _fast.FastPath(st)
    from .service import Service
    return Local(Service(st, fast=fp, routing=rt), model=model)


def load(*args, **kw):
    """horopter.runtime.load: the model as a Student (low-level answers; see local() for the API shape)."""
    try:
        from .runtime import load as _load
    except ImportError as e:
        raise ImportError(f"{_NEED} ({e})") from None
    return _load(*args, **kw)


def __getattr__(name):            # submodules and runtime names on first use, so the client never imports torch
    import importlib
    if name in _TORCH:
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError as e:
            raise ImportError(f"horopter.{name}: {_NEED} ({e})") from None
    if name in ("Student", "load_checkpoint"):
        return getattr(__getattr__("runtime"), name)
    raise AttributeError(f"module 'horopter' has no attribute {name!r}")
