# Adapters and routing

**Horopter serves the trunk alone unless you ask for adapters.** The trunk answers every question type with its own
heads, evidence spans and abstention head. Adapters specialise it: train them for your deployment's tasks, then train
a router over their descriptions (E51: a router trained on its own targets picks the right one 100% of the time; over
targets it never saw, 90–96%). The adapters bundled with the published models are a worked example of that recipe,
not a set every deployment needs:

| bundled adapter | what it is |
|---|---|
| `evidence`, `extract` | general helpers: evidence spans for every answer; typed field extraction |
| `check2` | a policy checker with an explicit "more information needed" outcome |
| `router` + `typed`, `nli`, `relation`, `classification`, `knowledge`, `clause` | the example: six task families and the router trained over them |

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
| `targets` | for a router: `{name: description}` of the adapters it was trained to choose between (same repo) |

## Where adapters come from

Name adapters with `--adapters` (server) or `adapters=[...]` (`horopter.local`). Only the adapters named are
downloaded, on first use; the hub cache keeps them after that. The trunk download never includes `adapters/` (0.8 GB
for 0.4B, 2.1 GB for 1B; each adapter adds 40 MB or 72 MB).

| reference | what it loads |
|---|---|
| `extract` | a bundled adapter: the model's own `adapters/extract/`, from the model's repo and revision |
| `./refunds` | a directory with `adapter.safetensors` (+ `adapter.json`); a parent directory loads every `<name>/` in it |
| `user/repo` | the adapter at a hub repo's root |
| `user/repo:refunds` | `adapters/refunds/` in a hub repo laid out like a model repo (else the folder `refunds/`) |
| `user/repo@v2:refunds` | the same at a revision (a tag, branch or commit) |
| `all` | every bundled adapter (the earlier default) |

The loaded name is `adapter.json`'s `name`, else the folder name; `--adapter NAME=REF` picks another. `--serve`,
`--router` and `--route` take the same references and load what they name, so `--router router` alone brings the
bundled router and the six adapters it routes between.

## Publishing adapters

Any hub repo can carry adapters for a Horopter model. Lay it out like the model repo:

```
someone/horopter-1B-retail
  README.md                    model card; front matter "base_model: jmccardle/horopter-1B" links it on the hub
  adapters/refunds/            adapter.safetensors + adapter.json
  adapters/triage/
  adapters/router/             a router trained over refunds and triage (optional)
```

- **One repo per trunk.** An adapter fits only the trunk it was trained on, and `adapter.json` `"p"` records it
  (`"prod_j1b_b3"`): the server refuses an adapter from another trunk. When the trunk changes (a new beta), retrain
  and publish under a new tag, and pin it: `someone/horopter-1B-retail@b3:refunds`.
- **Give every adapter a `description`** in `adapter.json`: a concrete one-line account of its questions. A router
  without `--route` flags routes over these.
- **A router lists its targets** in its `adapter.json`, `"targets": {"refunds": "a retail refund or return decision",
  "triage": "which support team gets a ticket"}`. The names are adapters in the same repo, so
  `--router someone/horopter-1B-retail:router` loads the router and both targets.
- Suggested name: `<owner>/horopter-<size>-<domain>`, with `base_model` in the card pointing at the Horopter model.

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

These numbers are from E51: picking a question's task among K one-line descriptions, with 504 questions over 42 tasks.

| router | K = 4 | K = 8 |
|---|---|---|
| a router trained over its own targets (shuffled, some removed) | 100% | 100% |
| bundled `router` over targets it never trained on (state layout) | 95–96% | 90–93% |
| `base`, no router adapter | 75–82% | 62–73% |

- **Best: train a router over your own adapters** (below). It takes minutes and is the only one that is right every time on its own targets.
- **To try targets out first, use the bundled router:** `--router router --route ...`. It learned the routing question, not just the six bundled answers.
- **`--router base` needs no router adapter**, but it is much weaker.
- **Write concrete descriptions** that say what the questions are about, such as "the sentiment of a financial news headline", not the kind of task ("labelling a text"). Over the six abstract bundled descriptions, `base` routes correctly only 17–53% of the time.

### Training a router for your adapters

```bash
python training/build_router_data.py --tokenizer jmccardle/horopter-1B --out data/router \
    --target refunds "a retail refund or return decision" data/refunds_train.jsonl \
    --target triage "which support team gets a ticket" data/triage_train.jsonl
python training/train_adapter.py --trunk <the model directory> --name router \
    --data data/router/router_train.jsonl --meta data/router/router_meta.json --steps 1500
```

- The rows are your adapters' own training rows. Each becomes "which of these targets should answer this?", asked in
  the `question` layout the server uses, with the options shuffled and some removed.
- `router_meta.json` puts `routing_layout` and `targets` into the router's `adapter.json`, so `--router ./router`
  needs no `--route` flags.
- Publish the router next to its targets (Publishing adapters, above).

### Fields and evidence

- Fields (Clairic extraction) are not routed. They use the `extract` adapter when one is loaded, else the served
  adapter (with routing: `classification` if loaded, else `base`). Load `extract` for extraction: `--adapters extract`.
- When an adapter named `evidence` is loaded, it supplies the evidence spans of every routed or served answer.
- To use your own, load it under that name: `--adapter evidence=./my_evidence`.

## Serving

| you want | flags |
|---|---|
| the trunk alone (the default) | none |
| the general helpers: evidence spans, field extraction | `--adapters evidence,extract` |
| one adapter, no routing | `--serve REF` |
| the bundled worked example (router over the six families) | `--router router` (add `--adapters evidence` for its spans) |
| your router over its own targets | `--router user/repo:router` |
| a router of your choice over targets of your choice | `--router REF --route REF [DESCRIPTION] --route ...` |
| zero-shot routing with no router adapter | `--router base --route ...` |

The default model name (`horopter`, or `--prefix`) is the routed model, or the served adapter (`base` by default).
`horopter-<adapter>` always answers with that one adapter. `/v1/models` and `/healthz` show the routing in use.

```bash
# the trunk alone: nothing but the model is downloaded
python -m horopter.server --model jmccardle/horopter-1B

# evidence spans and field extraction
python -m horopter.server --model jmccardle/horopter-1B --adapters evidence,extract

# one adapter of your own, no routing
python -m horopter.server --model jmccardle/horopter-0.4B --serve ./refunds

# your adapters and router, published in one repo
python -m horopter.server --model jmccardle/horopter-1B --router someone/horopter-1B-retail@b3:router

# two new adapters routed by their descriptions with the bundled router (--router base: no router adapter at all)
python -m horopter.server --model jmccardle/horopter-0.4B --router router \
    --route ./refunds "a retail refund or return decision" --route ./triage "which support team gets a ticket"
```

In Python, `horopter.local` takes the same references:

```python
import horopter
h = horopter.local("jmccardle/horopter-1B")                                  # the trunk alone
h = horopter.local("jmccardle/horopter-1B", adapters=["evidence", "extract"])
h = horopter.local("jmccardle/horopter-0.4B", serve="./refunds")
h = horopter.local("jmccardle/horopter-0.4B", router="base",
                   routes=[("./refunds", "a retail refund or return decision"), ("nli", None)])
```

With `"horopter": {"debug": true}` in a request, each answer's `horopter.route` shows the chosen adapter, its
probability and the router's full distribution (`question_cut: true` when the embedded question was shortened).
