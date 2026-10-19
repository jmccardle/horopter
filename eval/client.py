# SPDX-License-Identifier: MIT
"""Minimal HTTP client for a Horopter server (horopter/server.py, POST /v1/systemone).

The server URL comes from the environment: HOROPTER_URL, else MBNLI_URL (the research name), else
http://127.0.0.1:8420/v1/systemone. Model ids: "horopter" (router + adapters), "horopter-base" (trunk only),
"horopter-<adapter>"; the server also accepts its configured alias (by default "mbnli-3-beta[-...]").

    from client import ask
    out, ms = ask("The item was bought 3 days ago.", {"q": {"type": "noul", "instructions": "Is it recent?"}})
"""
import json
import os
import time
import urllib.request

URL = os.environ.get("HOROPTER_URL") or os.environ.get("MBNLI_URL") or "http://127.0.0.1:8420/v1/systemone"
DEFAULT_MODEL = os.environ.get("HOROPTER_MODEL", "horopter")


def ask(state, questions, model=None, evidence=None, url=None, timeout=300):
    """One request: every question about `state` in the same body. evidence = per-token threshold (mode "all"),
    or None for no evidence. -> (response body, milliseconds)."""
    body = {"model": model or DEFAULT_MODEL, "state": state, "questions": questions,
            "mbnli": {"allow_truncation": True}}
    if evidence is not None:
        body["mbnli"]["evidence"] = {"mode": "all", "threshold": evidence}
    req = urllib.request.Request(url or URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    return out, 1000 * (time.perf_counter() - t)
