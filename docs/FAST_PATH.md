# Fast path (`horopter.fast`, `python -m horopter.server --fast`)

The fast path serves the same answers as the release path, up to bf16 rounding on a GPU and exactly on a CPU. On a GPU it is 3–19× faster; on a CPU it is 2–26× faster for a state that has been seen before. Measured 2026-10-21.

## What it does

Horopter encodes the state without seeing any question. Every adapter's LoRA is gated off on state tokens. So a request splits in two:

- a **state pass**, cached by the state's token ids;
- a **question pass** that reads the cached keys and values.

The fast path builds on that split:

| piece | module | effect |
|---|---|---|
| state cache | `stateq`, `graphq` | A state asked about again costs only its question tokens. Up to `--slots` states (default 3) stay cached, least recently used out. |
| flat adapter bank | `bank` | An adapter swap is one copy per dtype instead of one per tensor. |
| CUDA graphs + `torch.compile` | `graphq` | Each pass replays as one graph per padded shape, so the CPU no longer launches kernels one at a time. |
| block-sparse state attention | `flexattn` | FlexAttention on the sliding-window layers of the state pass. It pays from about 500 state tokens. |
| all adapters in one pass | `multilora`, `vheads` | Every adapter of a request runs in one question pass (segmented LoRA). The heads and the answer maths run in the same graph, followed by one device-to-host copy. |

**On CUDA:**
- All five pieces are on.
- The trunk and heads run with bf16 weights; the answer head stays fp32.

**On CPU:**
- Only the state cache and `torch.compile` are on, in fp32.
- A CPU is compute-bound, so graphs have nothing to remove.
- The one-pass multi-adapter path is slower there, because its padding costs real arithmetic.

**Requirements:** torch ≥ 2.5 for the Flex state pass (tested with 2.9.1 on an RTX 4090 and 2.11 on a Jetson Orin). `FastPath(st, flex=False)` runs without it.

**Not accelerated:** the router pass. A routed request with *n* questions runs *n* router packs (question + the first 1,500 characters of the state) through the full forward. On a 4090 this costs about 40 ms for 4 questions. Name the adapter (`horopter-clause`, `horopter-nli`, …) to skip it.

## Measurements

All latencies are p50 in ms.
- **warm:** the state was sent before.
- **cold:** a new state.

### RTX 4090, in process

Release path:
- one full forward per adapter group;
- fp32 weights with bf16 autocast;
- no cache.

Workloads:
- **simple:** 2 yes/no questions on the base model;
- **one adapter:** 4 questions (yes/no, choice, score);
- **four adapters:** the same 4 questions, each on a different adapter.

| workload | state tokens | 0.4B release | 0.4B fast, warm / cold | 1B release | 1B fast, warm / cold |
|---|---|---|---|---|---|
| simple | 150 | 19 | 5.0 / 8.6 | 19 | 6.8 / 12.6 |
| simple | 2,000 | 44 | 9.9 / 27.5 | 83 | 12.2 / 49.7 |
| one adapter | 500 | 20 | 7.1 / 13.0 | 32 | 8.7 / 19.1 |
| four adapters | 150 | 80 | 7.5 / 11.3 | 79 | 11.4 / 17.1 |
| four adapters | 500 | 80 | 8.4 / 14.4 | 109 | 12.5 / 22.8 |
| four adapters | 2,000 | 167 | 12.6 / 30.1 | 321 | 17.2 / 54.7 |

### RTX 4090, over HTTP

Horopter-1B, `--fast --precapture default`. Times are the client's wall time. The last column is the server it replaced: the research server the release server was ported from, with the same architecture and size but the earlier beta weights. It has no state cache, so warm equals cold there.

| request | ~150 tok, warm / cold | ~500 tok, warm / cold | ~2,000 tok, warm / cold | previous server, ~150 / 500 / 2,000 tok |
|---|---|---|---|---|
| routed, 1 question | 22 / 34 | 18 / 35 | 35 / 77 | 60 / 55 / 124 |
| routed, 4 questions | 52 / 58 | 64 / 74 | 86 / 117 | 88 / 113 / 181 |
| one adapter, 4 questions | 11.5 / 17 | 14 / 24 | 28 / 58 | 27 / 43 / 79 |
| routed, 4 questions + evidence | 54 / 59 | 73 / 82 | 79 / 116 | 131 / 184 / 253 |

**Many questions per request** (0.0.5; Horopter-1B beta-3, `--fast --slots 8 --precapture default`, a 727-token agent tool log, `horopter-nli`, yes/no questions; median server ms, cold = a new state; research repo `experiments/v3/e54_arecibo_serve/bench005.py`). The server answers in passes of `--max_batch` (default 32) questions over the cached state, so cost per question stays flat; 0.0.4 answered in one pass and ran out of memory with evidence from about 150 questions.

| questions | evidence off, warm / cold | 0.0.4, warm / cold | evidence chosen, warm / cold | 0.0.4, warm / cold |
|---|---|---|---|---|
| 10 | 21 / 169 | 21–24 / 162–253 | 48 / 280 | 40 / 171–242 |
| 50 | 109 / 508 | 124–147 / 483–628 | 284 / 1,208 | 315 / 327–1,257 |
| 100 | | | 603 / 1,983 | 1,015 / 4,072 |
| 150 | | | 899 / 2,231 | out of memory |
| 200 | 519 / 1,271 | 1,001 / 3,921 | 1,208 / 1,706 | out of memory |
| 400 | | | 2,392 / 2,684 | out of memory |

- About 2.2–2.6 ms per question warm without evidence and 6 ms with it (evidence costs 2–3×).
- Cold times vary 2–4× between runs: a new state, plus a graph capture for each pass shape not precaptured (the default shapes cover up to 4 questions per adapter). A deployment with a fixed request shape can precapture it: `FastPath.discover(requests)`.
- GPU memory stayed at about 14.4 GB through 400 questions with evidence (22 GB at 100 in one pass on 0.0.4).

### Jetson Orin Nano 8 GB, in process

- **Precision:** bf16, compiled and graphed.
- **Two adapters:** questions 1 and 3 on the clause adapter, 2 and 4 on nli.
- **Cold** is dominated by the state pass: about 30 ms at 150 tokens, 68 ms at 500, and 317 ms at 2,000 (with Flex).

| workload | 150 tok, warm / cold | 500 tok, warm / cold | 2,000 tok, warm / cold |
|---|---|---|---|
| simple | 30 / 61 | 35 / 104 | 53 / 375 |
| one adapter, 4 questions | 50 / 81 | 55 / 125 | 83 / 406 |
| two adapters, 4 questions | 59 / 90 | 65 / 135 | 94 / 416 |
| four adapters, 4 questions | 73 / 105 | 82 / 151 | 118 / 441 |

**Robotics:** send the state ahead of the question (`FastPath.put_state`) to move the state pass out of the critical path.

### CPU (i5-12400, 6 threads, fp32, 0.4B)

| workload | state tokens | release | fast (state cache + compile), warm / cold |
|---|---|---|---|
| simple | 150 | 338 | 152 / 356 |
| simple | 2,000 | 4,466 | 246 / 3,838 |
| one adapter | 500 | 1,093 | 317 / 981 |
| four adapters | 500 | 3,388 | 440 / 1,109 |
| four adapters | 2,000 | 17,315 | 673 / 4,051 |

Two CPU options were tried and are not recommended:
- **torch's dynamic int8 kernels:** 1.4× faster, but they changed 16 of 30 answers, because they use one activation scale per tensor.
- **The Q8 / Q4 files:** these are dequantized at load. They make a smaller download but give no speed or memory saving.

## Memory and start-up

| | 0.4B | 1B |
|---|---|---|
| GPU memory, 3 slots | 3.0–3.2 GB reserved (4090, the benchmark's shapes captured) | 8.5 GB (4090, process total, all default shapes captured) |
| `--precapture default`, start-up (81 shapes × 3 slots = 243 graphs) | 11–16 s (4090, measured with 2–3 slots) | 286 s on a first start (empty torch compile cache, other jobs on the GPU), 66 s after (4090) |

**Precapture.** Graphs are captured the first time a shape is seen. A shape is (padded state length, question tokens, adapters per request, options, evidence mode).
- Without `--precapture`, the first request of each new shape pays its capture: tens to hundreds of ms.
- With `--precapture default`, all 81 default shapes are captured for every slot at start-up.
- The router pass is compiled with dynamic shapes, so the first routed requests also pay a compile (about 2 s once).
- Captured graphs and the state slots stay resident. On a shared-memory device (Jetson), capture only the shapes the deployment uses: list them with `FastPath.discover(requests)` and pass the list as `precapture`.

**CPU memory:** each cached 2,000-token state holds about 0.46 GB of fp32 keys and values. Set `cache_size` to what the host can spare.

## Correctness

`tests/test_fast.py` (`HOROPTER_TEST_MODEL=path python -m pytest -s tests/test_fast.py`):

- **cpu:**
  - served answers through the endpoint, compared with the release path over 4 states and 2 cache slots in rotating order (hits, misses and evictions);
  - raw probabilities and evidence scores: max |dp| 1.3e-6 on the 0.4B;
  - evidence spans identical.
- **cuda:**
  - compared with the release path: max |dp| 0.009 on the 0.4B and 0.006 on the 1B; an answer may differ only on a tie;
  - replay check: 8 states of 40–1,900 tokens against 3 graph slots, rotating, warm, then after a cache reset, then warm again. Every answer equals its first answer (drift 0).

**Rule for anyone changing this code:** every tensor a captured graph reads must be held for the graph's whole lifetime. An earlier version freed a rotary-position tensor after capture, and later replays read the freed memory. Test changes with eviction and repeated replays through the served path, not with a single first capture.

## Using it from Python

```python
from horopter import runtime, fast
st = runtime.load("path/to/horopter-0.4B", device="cuda")
fp = fast.FastPath(st)                               # takes over st: use st only through fp.call from now on
fp.call(lambda: fp.put_state(state))                 # optional: encode ahead of the question
route = {"q": {"adapter": "clause"}}
res = fp.call(lambda: fp.answer(state, {"q": question}, route, evidence=False))
```

`fp.answer` returns the dicts `Student.answer` returns. The wire format (typed answers, relation, abstention, evidence spans) is built from them by `horopter.server`.
