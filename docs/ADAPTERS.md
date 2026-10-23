# Adapters and routing

## What an adapter is

An **adapter** is a small file set that changes how Horopter answers questions, on top of one fixed trunk:

```
my_adapter/
  adapter.safetensors   LoRA weights (rank 16) on the question path + a full copy of the answer heads (~38 MB on 0.4B)
  adapter.json          metadata: name, the trunk it was trained on ("p"), training data and dev accuracy
```

- The LoRA only touches **question tokens**. The state (the text or JSON you ask about) is encoded the same way
  whatever adapter is active, which is why one cached state serves every adapter (docs/FAST_PATH.md).
- Every adapter answers every question type (`noul`, `choice`, `score`, fields). An adapter is a specialist, not a
  single head.
- **An adapter fits only the trunk it was trained on.** `adapter.json` records it as `"p"` (e.g. `"prod_e400m_b2"`);
  loading an adapter from another trunk is refused (the server's `--allow_trunk_mismatch` overrides this, and the
  answers will be wrong).
- `base` is the trunk with no adapter. It answers everything too.

Optional `adapter.json` fields used by routing:

| field | meaning |
|---|---|
| `description` | one line saying what the adapter is for; the default routing description |
| `routing_layout` | for an adapter trained as a router: the layout it was trained on (`question` or `state`) |

## Where adapters come from

`--adapter [NAME=]SOURCE` (repeatable) or `horopter.local(adapters_from=...)`. SOURCE is

- a directory: `./refunds`
- a hub repository with the adapter at its root: `user/horopter-refunds`
- a folder inside a hub repository: `jmccardle/horopter-0.4B:adapters/nli`

NAME defaults to the `name` in `adapter.json`, else the directory name. The model's own `adapters/` are loaded too
unless `--no_adapters` is given.

## Routing

A **router** is any adapter, or `base`, asked one hidden `choice` question per routed question:

```
instructions  "Which kind of task does this question most resemble?" + the routed question
criteria      {target adapter name: its description, ...}
```

The top option names the adapter that answers. Nothing else about routing is built into the server: the targets are
the options, their descriptions are the option texts, and the same question can be sent to any model through the
normal API.

The bundled `router` adapter was trained on this question over the six bundled family adapters (`typed`, `nli`,
`relation`, `classification`, `knowledge`, `clause`).

### Layouts: what the router reads

| layout | the router reads | cost |
|---|---|---|
| `question` | the request's own state; the routed question is embedded in the router question's instructions, shortened if needed so every option still fits the 128-token question block | no extra state pass: the state is shared with the answer pass |
| `state` | a fresh state per routed question: the question's type, instructions and option names, then the first 1,500 characters of the state | one extra state pass per routed question |

The bundled router was trained on the `state` layout and uses it by default. Any other router uses `question` unless
its `adapter.json` says otherwise, or `--route_layout` is given.

### Choosing a router

These numbers are from E51: picking a question's task among K one-line descriptions the router never trained on, with 504 questions over 42 tasks.

| router | K = 4 | K = 8 |
|---|---|---|
| bundled `router` (state layout) | 95–96% | 90–93% |
| `base`, no router adapter | 75–82% | 62–73% |

- **Use the bundled router for your own targets too:** `--router router --route ...`. It learned the routing question, not just the six bundled answers.
- **`--router base` needs no router adapter**, but it is much weaker.
- **Write concrete descriptions** that say what the questions are about, such as "the sentiment of a financial news headline", not the kind of task ("labelling a text"). Over the six abstract bundled descriptions, `base` routes correctly only 17–53% of the time.

### Fields and evidence

- Fields (Clairic extraction) are not routed. They use the `extract` adapter when one is loaded, else the served
  adapter (with routing: `classification` if loaded, else `base`).
- When an adapter named `evidence` is loaded, it supplies the evidence spans of every routed or served answer.
- To use your own, load it under that name: `--adapter evidence=./my_evidence`.

## Serving

| you want | flags |
|---|---|
| the bundled setup (bundled router over the bundled families) | none |
| one adapter, no routing | `--serve NAME` (`--serve base`: the trunk alone) |
| a router of your choice over targets of your choice | `--router NAME --route NAME [DESCRIPTION] --route ...` |
| zero-shot routing with no router adapter | `--router base --route ...` |

The default model name (`horopter`, or `--prefix`) is the routed model, or the served adapter with `--serve`.
`horopter-<adapter>` always answers with that one adapter. `/v1/models` and `/healthz` show the routing in use.

```bash
# one adapter, no routing
python -m horopter.server --model jmccardle/horopter-0.4B --no_adapters --adapter refunds=./refunds --serve refunds

# two new adapters, routed by their descriptions with the bundled router (--router base: no router adapter at all)
python -m horopter.server --model jmccardle/horopter-0.4B \
    --adapter ./refunds --adapter ./triage --router router \
    --route refunds "a retail refund or return decision" --route triage "which support team gets a ticket"

# one new adapter added to the bundled router's options, with the competing bundled one removed
python -m horopter.server --model jmccardle/horopter-0.4B --adapter ./refunds \
    --route typed --route nli --route relation --route knowledge --route refunds "a retail refund or return decision"
```

In Python:

```python
import horopter
h = horopter.local("jmccardle/horopter-0.4B", adapters=False, adapters_from={"refunds": "./refunds"},
                   serve="refunds")
h = horopter.local("jmccardle/horopter-0.4B", adapters_from=["./refunds"], router="base",
                   routes=[("refunds", "a retail refund or return decision"), ("nli", None)])
```

With `"horopter": {"debug": true}` in a request, each answer's `horopter.route` shows the chosen adapter, its
probability and the router's full distribution (`question_cut: true` when the embedded question was shortened).
