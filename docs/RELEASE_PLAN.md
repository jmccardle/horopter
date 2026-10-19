# Horopter beta 2: release plan (drafted 2026-10-18)

**Targets** (all three exist, are empty, and are **public**, so a push publishes immediately):
- https://github.com/jmccardle/horopter: code, training scripts, data recipe, docs. Local: `/storage/horopter`, remote `origin` set, **not pushed**.
- https://huggingface.co/jmccardle/horopter-0.4B: weights (Ettin-encoder-400M base). Local draft: `/storage/horopter-hf/horopter-0.4B/`.
- https://huggingface.co/jmccardle/horopter-1B: weights (Ettin-encoder-1B base). Local draft: `/storage/horopter-hf/horopter-1B/`.

**Name.** Horopter is the internal "mbnli-3" model (research repo `/storage/ModernBERT-NLI-advanced`, ROADMAP.md).
- In optics, the horopter is the set of points the two eyes see as one.
- The model fuses two views, a cached question-blind state and the question, into one answer, and reports where they agree (relation, evidence) and where they don't (abstention).
- Theme art: the title page of *Opticae thesaurus* (Risner, Basel 1572; Ibn al-Haytham's optics with Witelo's), public domain. It is in `assets/`: rainbow, sun rays reflected by mirrors onto the ships at Syracuse, a man studying his reflection, and a bather in the water.

## Timeline (48 h from 2026-10-18 evening)

| when | GPU (4090) | other |
|---|---|---|
| now → about 02:00 | **E39: prod_e400m_b2** (400M, beta-2 data) training | scaffold this repo; port the model code |
| about 02:00 → 04:00 | E37 bench + real policy items + generated policies on the 400M b2 trunk | read out: 400M b2 vs prod_e400m |
| day 1 morning | **1B beta-2 run** (prod_j1b recipe + prod2 data, about 10–11 h). :8420 is down for it, or served from the 400M meanwhile | export scripts, model cards, data recipe |
| day 1 evening | 400M: adapters (router, families, evidence, check2) + E35 abstention head refit + E37 | |
| day 2 | 1B: adapters + head + E37 + E40; final tables | release candidates; the user reviews; push |

The 400M result gates the 1B run. If the beta-2 data costs grounding (AggreFact / E0 drop more than 1 pt), the recipe is fixed first.

## What ships

1. **Weights** (each HF repo):
   - `model.safetensors`: the trunk with its answer, relation and evidence heads;
   - `adapters/`: router, six family LoRAs, evidence, check2;
   - `abstain_head.json`: the E35 logistic head;
   - `config.json`, tokenizer from Ettin.
2. **Code** (GitHub):
   - `horopter/`: model, packing, adapters, server with the wire format;
   - `training/`: the joint trainer, adapter and evidence trainers;
   - `eval/`: E37 bench scorer, E40 rule evals.

   This is a curated port of `experiments/v3/e12_joint`, `e24_dryrun_router`, `e35_abstention_head` and `e39_beta2_data`, not the research monorepo.
3. **Data recipe** (`training/recipes/`):
   - the prod2 manifest (pools, caps, repeats, mix, temperatures);
   - the builders (policy generator, check-v2 generator, counterfactual pipeline, prod2 builder);
   - per-source licences and the `permissions` column (RELEASE_POLICY.md §2).

   Source texts that are share-alike or not redistributable ship as pointers (§3). LLM-AggreFact is never included (eval only).
4. **Reports:** the beta and beta-2 benchmark tables (E37), the effort/interpreter probes (E38), and the rule evals (E40).

## Decisions needed (the user)

- **R1, licence** for code and weights: MIT (matches Ettin, recommended) / Apache-2.0 / other. RELEASE_POLICY §6 leaves this to counsel.
- **R2, one model or two** (RELEASE_POLICY §5):
  - Beta 2 trains on the max data, which includes `restricted` (NC / research-only) sources and Jev labels.
  - Releasing it under a permissive licence is the open counsel item.
  - Options: (a) release max models with the Jev + restricted notice; (b) release only an open variant (libre + jev), which means extra training runs; (c) max under a non-commercial licence.
- **R3, data release scope at launch:** recipe + scripts + our synthetic data (generated policies, check-v2, counterfactual pairs: all libre) now, and the full committee dataset (with Jev outputs, pointers for SA sources) later (recommended) / everything at once.
- **R4, served names:** rename the API model ids `mbnli-3-beta-*` to `horopter-*`, keeping the old names as aliases (recommended).
- **R5, adapters in the HF repos:** yes, under `adapters/` (recommended) / separate repos.

## Checklist before any push

- [ ] Weights load from the HF layout with `horopter/` alone (fresh venv, CPU smoke + GPU).
- [ ] Model cards: intended use, wire format, evidence-first framing, benchmark tables, limits (multi-step policy gap, "more information" never chosen), training data summary with licences, Jev notice, restricted-source notice.
- [ ] No eval-only data in the repo (AggreFact, JevBench/JF100 items, ConditionalQA test).
- [ ] No secrets, keys, hostnames or internal paths in the code (`/storage/...`, Xavier IPs, TypeSafe keys).
- [ ] Licence files: the code licence, the weights licence, the image's public-domain note.
- [ ] The user signs off on the cards and the README.
