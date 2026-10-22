# Horopter

<p align="center"><img src="assets/thesaurus_opticus_1572_titlepage.jpg" width="420" alt="Title page of Opticae thesaurus (Basel, 1572)"></p>

> *Horopter*: the set of points that both eyes see as one.

Code: MIT (`LICENSE`). Weights, data recipe and docs: CC BY-NC-SA 4.0 (research, non-commercial). Weights: [horopter-0.4B](https://huggingface.co/jmccardle/horopter-0.4B), [horopter-1B](https://huggingface.co/jmccardle/horopter-1B) (first release, 2026-10-20).

Horopter is an encoder-only *decision model*. Give it a **state** (a document, a record or a JSON object) and typed **questions** about it (yes/no, choose one, score on a scale). It answers them in one pass, and for every answer it also returns:

- **evidence:** whole-word spans in the state that support or refute each option;
- **relation:** whether the state bears on the question at all, and with what stance (support / refute / undetermined);
- **abstention:** a calibrated probability that the answer is wrong, with a reason (`unrelated`, `model_uncertainty`).

The state is encoded once, without seeing any question, and cached. Each question reads that cached state, so:
- adding a question never changes the answers to the others;
- asking many questions about one state costs about one encoding.

Built on [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-1b) encoders (400M and 1B), from Fight Fire with Fire Robotics.

## Status

First release, **2026-10-20**: [horopter-0.4B](https://huggingface.co/jmccardle/horopter-0.4B) (`horopter-0.4b-20261020`) and [horopter-1B](https://huggingface.co/jmccardle/horopter-1B) (`horopter-1b-20261020`). Router + adapters, one pass:

| | Horopter 1B | Horopter 0.4B | mbnli-3 beta 1B (internal) | best 4–9B decoder we tested |
|---|---|---|---|---|
| grounding (LLM-AggreFact balanced accuracy) | 75.7 | 74.8 | 75.7 | 77.2 |
| 4-way relation (macro-F1) | 78.6 | 76.6 | 78.6 | 75.3 |
| multi-step policy decisions (JevBench hard) | 40.5 | 36.0 | 44.1 | 64.0 |
| generated policies, one pass (prose) | 97.7 | 95.0 | 53.6 | |

The full tables and limits are in the model cards. Grounding and relation are level with 4–9B decoders. Multi-step policy decisions on real text trail them by about 20 points. This release has rule-application data with always-determined answers, so it rarely says "more information needed" on real rule text.

## Layout

```
horopter/     model, packing, losses, runtime (adapters), evidence, abstention, server (typed wire format)
horopter/fast/  fast path: state cache, CUDA graphs, block-sparse state attention, all adapters in one pass
training/     joint trainer, adapter / evidence trainers, abstention head fit, data builders, recipes/beta2.md
eval/         benchmark scorers (E37 bench, E40 rule evals) and the HTTP client
scripts/      export_hf.py: training checkpoint -> HF layout
tests/        parity against the research code; fast path against the release path
docs/         release plan, porting notes (docs/PORTING.md), fast path (docs/FAST_PATH.md)
assets/       theme art (public domain)
```

```python
import horopter
m = horopter.load("path/to/horopter-0.4B")      # the exported HF layout
out = m.answer("Refunds within 30 days. The lamp was bought 12 days ago.",
               {"q": {"type": "noul", "instructions": "Is a refund allowed?"}}, evidence=True)
```

Server: `python -m horopter.server --model path/to/horopter-0.4B --port 8420 --fast`.

`--fast` serves the same answers (up to bf16 rounding) 3–19× faster on a GPU, and caches states on a CPU: a state asked
about again costs only its question tokens. Measurements, settings and memory are in [docs/FAST_PATH.md](docs/FAST_PATH.md).

## Credits

Title page of *Opticae thesaurus* (Friedrich Risner ed., Basel 1572), Bayerische Staatsbibliothek; public domain (Wikimedia Commons, `Thesaurus_opticus_Titelblatt.jpg`).
