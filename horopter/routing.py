# SPDX-License-Identifier: MIT
"""Who answers a question: one adapter directly, or a router's choice among target adapters.

A router is any adapter (or "base", the trunk with no adapter) asked one choice question per routed question:
    instructions  "Which kind of task does this question most resemble?" + the routed question
    criteria      {target adapter name: its description}, in the order given
The top option names the adapter that answers. Nothing else about routing is built in.

Layouts (what the router reads):
    question  the request's own state, with the routed question embedded in the router question (cut so every option
              block still fits the 128-token question block). The state pass is shared with the answer pass.
    state     a fresh state per routed question, families.router_state(state, q): the question + the first 1,500
              characters of the state. The layout the bundled router was trained on (E24); one extra state pass each.

    from horopter import routing
    rt = routing.configure(st, router="base", routes=[("refunds", "a retail refund decision"), ("triage", None)])
    rt = routing.configure(st, serve="nli")          # no routing: every question goes to the nli adapter
"""
import json

from . import families as F

ASK = F.ROUTER_Q["instructions"]
HELPERS = ("router", "evidence", "extract")      # bundled helper adapters: never routing targets by default


class RoutingError(ValueError):
    pass


def describe(st, name):
    """A target's description: adapter.json "description", else the bundled family description, else its name."""
    return (st.meta.get(name) or {}).get("description") or (F.FAMILIES[name][1] if name in F.FAMILIES else name)


def _opts(q):
    crit = q.get("criteria")
    if isinstance(crit, dict):
        return " | ".join(map(str, crit))
    if isinstance(crit, list):
        return " | ".join(f"level {i}" for i in range(len(crit)))
    return "true | false"


class Routing:
    """serve: answer every question with this adapter (no router). Otherwise router + targets {name: description}."""

    def __init__(self, serve=None, router=None, targets=None, instructions=ASK, layout="question"):
        if (serve is None) == (router is None):
            raise RoutingError("give exactly one of serve (one adapter, no routing) or router")
        if layout not in ("question", "state"):
            raise RoutingError(f"layout must be question or state, got {layout!r}")
        self.serve, self.router, self.instructions, self.layout = serve, router, instructions, layout
        self.targets = dict(targets or {})
        if router is not None and len(self.targets) < 2:
            raise RoutingError("a router needs at least 2 targets; to answer with one adapter, serve it directly")

    @property
    def routed(self):
        return self.router is not None

    def question(self, q, tok, max_q=128):
        """The router question about q. -> (question, cut): cut is True when the embedded question was shortened."""
        crit = dict(self.targets)
        if self.layout == "state":
            return {"type": "choice", "instructions": self.instructions, "criteria": crit}, False
        instr = q.get("instructions")
        instr = instr if isinstance(instr, str) else json.dumps(instr, sort_keys=True)
        full = f"{self.instructions} Question ({q['type']}): {instr} Options: {_opts(q)}"
        # every option block is "<instructions> Answer: <name>: <description>", cut at max_q - 1 tokens
        longest = max(len(tok(f" Answer: {k.replace('_', ' ')}: {d}", add_special_tokens=False)["input_ids"])
                      for k, d in crit.items())
        ids = tok(full, add_special_tokens=False)["input_ids"]
        budget = max(8, max_q - 1 - longest)
        if len(ids) <= budget:
            return {"type": "choice", "instructions": full, "criteria": crit}, False
        return {"type": "choice", "instructions": tok.decode(ids[:budget]), "criteria": crit}, True

    def info(self):
        if not self.routed:
            return {"serve": self.serve}
        return {"router": self.router, "layout": self.layout, "instructions": self.instructions,
                "targets": dict(self.targets)}


def configure(st, serve=None, router=None, routes=None, instructions=None, layout=None):
    """A Routing for the adapters loaded in st.bank.
    serve: one adapter name (or "base"): no routing.
    router: an adapter name (or "base"); routes: [(name, description or None)] targets (default: the bundled families
    that are loaded). layout default: the router's adapter.json "routing_layout", else "state" for the bundled router
    (its training layout), else "question".
    With neither serve nor router: the bundled router over the bundled families if a "router" adapter is loaded,
    else base served directly."""
    bank = st.bank
    if serve is not None and (router is not None or routes):
        raise RoutingError("serve answers with one adapter; it cannot be combined with router or routes")
    if serve is None and router is None:
        if "router" not in bank:
            if routes:
                raise RoutingError("routes need a router: pass router (any loaded adapter, or base)")
            serve = "base"
        else:
            router = "router"
    for name in [serve, router] + [r[0] for r in routes or []]:
        if name is not None and name not in bank:
            raise RoutingError(f"adapter {name!r} is not loaded; loaded: {sorted(bank)}")
    if serve is not None:
        return Routing(serve=serve)
    if routes:
        targets = {}
        for name, desc in routes:
            if name in targets:
                raise RoutingError(f"route {name!r} given twice")
            targets[name] = desc or describe(st, name)
    else:
        targets = {f: describe(st, f) for f in F.FAMILIES if f in bank}
    if layout is None:   # an adapter.json may say which layout its router was trained on
        layout = (st.meta.get(router) or {}).get("routing_layout") or ("state" if router == "router" else "question")
    return Routing(router=router, targets=targets, instructions=instructions or ASK, layout=layout)
