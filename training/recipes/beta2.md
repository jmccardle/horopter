# Beta-2 training recipe (horopter-0.4B, horopter-1B)

This is the exact recipe of the beta-2 runs. In the research repo they were `prod_e400m_b2` (400M) and
`prod_j1b_b2` (1B). The commands below were copied from `retrain_b2_400m.sh` and `gate_then_1b.sh` and rewritten for
this repo's CLI. Only paths changed; the arguments are the same. The two sizes use identical arguments except
`--base`.

**No data ships with this repo.** Every file under `data/` must be supplied by you, under `$HOROPTER_DATA`. That
covers pools, cached teacher answers, targets, lengths, roles, conflict pairs, the E15 drop lists, the eval sets and the
E37 bench. The list is in [What is not shipped](#what-is-not-shipped). `prod2_manifest.json` (next to this file) is the
metadata of the beta-2 build: pools, caps, temperatures, stacking weights, counts. It contains no data.

## Environment

```bash
export HOROPTER_DATA=/path/to/data-root      # holds data/v3/... (the layout the research build used)
export HOROPTER_OUTPUTS=/path/to/outputs     # checkpoints, adapters, reports
export HF_HOME=/path/to/hf-cache             # Ettin weights and tokenizer (jhu-clsp/ettin-encoder-400m, -1b)
pip install -e '.[training]'
```

Relative data paths in the commands (for example `data/v3/prod/eval_a25_test.jsonl`) resolve against `--data_root`.
Its default is `$HOROPTER_DATA`, or the current directory.

## Pipeline

```
E34 build (lengths.py, build.py)  ->  data/v3/prod             the beta's typed targets + a25 evidence stream
beta-2 streams (policy_gen, node_data, check_v2_data, cf_pairs) -> data/v3/e38/*
build_prod2.py                    ->  data/v3/prod2 + data/v3/e39/pool_e39.jsonl
train_joint.py                    ->  $HOROPTER_OUTPUTS/e12/<name>/model.pt
scripts/export_hf.py              ->  HF layout (trunk only)
train_adapter.py / train_evidence.py on the exported trunk -> adapters
fit_abstain_head.py               ->  abstain_head.json
scripts/export_hf.py again        ->  HF layout with adapters/ and abstain_head.json
eval/ (E37 bench, E40 rule evals) against horopter/server.py
```

### 1. E34 production data (the beta's mix; CPU, about 4 min + 4 min)

```bash
python training/data/lengths.py            # token lengths of every pool row (once per pool change)
python training/data/build.py --caps training/data/caps.json \
    --drop_premises $HOROPTER_DATA/data/v3/e15/snli_heldout_image_premises.json \
    --drop_rows $HOROPTER_DATA/data/v3/e15/minhash_drop_narrow.json
```

The build works in this order:
1. It reads the pools `train`, `a23`, `a23anli` (`data/v3/e09/pool_*.jsonl`) and `a25` (`data/v3/e33/pool_a25.jsonl`).
2. It reads the cached teacher answers `data/v3/e09/raw/<teacher>.<pool>.jsonl`, for Jev, decider and clef-flash.
3. It freezes the a25 held-out split (`eval_a25_split.json`; if the file exists it is reused).
4. It calibrates each teacher, then fits the per-cluster stack.
5. It writes `data/v3/prod/{targets,lengths}/`, `evidence_a25.jsonl.gz`, `eval_a25_{dev,test}.jsonl` and
   `manifest.json`.

Re-running it on the research inputs reproduces the research `data/v3/prod` byte for byte. This was checked for the
targets, the evidence, the eval split and the manifest; only the `built` timestamp differs.

### 2. Beta-2 streams (CPU; the LLM steps need a self-hosted OpenAI-compatible server)

```bash
D=$HOROPTER_DATA/data/v3/e38
python training/data/policy_gen.py --n 600 --seed 0 > $D/policy_v0.jsonl        # E38b eval set: kept OUT of training
python training/data/node_data.py --n 6000 --seed 1 --holdout robot_sop > $D/check_train.jsonl   # check adapter v1

# check v2 (Qwen3-30B-A3B writes policies + hard-negative conditions; a blind second pass verifies them)
python training/data/check_v2_data.py build --n 300     # as the research run (its ck2_jobs has 300 rows; the script default is 600)
python training/tools/llm_fanout.py run --endpoints endpoints.json --no-think --temperature 0.8 \
    --in $D/ck2_jobs.jsonl --out $D/ck2_gen.jsonl
python training/data/check_v2_data.py verify
python training/tools/llm_fanout.py run --endpoints endpoints.json --no-think --in $D/ck2_verify_jobs.jsonl --out $D/ck2_verify.jsonl
python training/data/check_v2_data.py assemble --bench /path/to/e37/bench.jsonl   # -> check_train_v2.jsonl, check_dev_v2.jsonl

# counterfactual minimal pairs: v1 (Gemma-4-E4B edits) and v2 (Qwen3-30B-A3B, 3 proposals, blind-verified)
python training/data/cf_pairs.py build --n 600
python training/tools/llm_fanout.py run --endpoints endpoints.json --in $D/cf_jobs.jsonl --out $D/cf_edits.jsonl
python training/data/cf_pairs.py verify
python training/tools/llm_fanout.py run --endpoints endpoints.json --in $D/cf_check_jobs_v2.jsonl --out $D/cf_checks_v3.jsonl
python training/data/cf_pairs.py score          # needs a server (HOROPTER_URL): writes cf_scored.jsonl (v1 validity)
python training/data/cf_pairs.py build2         # --per 70
python training/tools/llm_fanout.py run --endpoints endpoints.json --in $D/cf2_jobs.jsonl --out $D/cf2_edits.jsonl
python training/data/cf_pairs.py verify2
python training/tools/llm_fanout.py run --endpoints endpoints.json --in $D/cf2_check_jobs.jsonl --out $D/cf2_checks.jsonl
python training/data/cf_pairs.py select2        # -> cf2_pairs.jsonl
```

The LLM settings in these commands, other than `--no-think` and `--temperature`, are the job files' own, set in
`params` per row (`response_format`). The model behind each endpoint is yours to choose. The provenance of the
research data:
- Gemma-4-E4B wrote the v1 counterfactual edits.
- Qwen3-30B-A3B wrote the v2 pairs and the check-v2 policies, and ran the blind verification.

`cf_pairs.py score` asks the beta trunk on both texts. Only its `valid` flag, which comes from the judge, feeds
build_prod2.

### 3. prod2 (CPU, about 15 s)

```bash
python training/data/build_prod2.py --bench /path/to/e37/bench.jsonl      # defaults: --n_policy 6000 --n_policy_ev 3000
                                                                          #   --k_policy 3 --k_check 3 --k_cf 4 --seed 3
```

The output is `data/v3/prod2`: links to the prod targets, lengths and eval files, plus `targets/e39.jsonl.gz`,
`lengths/e39.json.gz`, `evidence_prod2.jsonl.gz` and `manifest.json`. It also writes `data/v3/e39/pool_e39.jsonl`.
`--bench` (the E37 bench) and `--policy_eval` (default `data/v3/e38/policy_v0.jsonl`) are used for
decontamination only. Rows whose state shares a 13-gram with the bench are dropped, and policy_v0's eval states are
never generated into training.

Re-running it on the research inputs reproduces the research `prod2` e39 pool, targets, lengths and evidence byte for
byte.

### 4. Joint training (GPU)

**400M** (`prod_e400m_b2` = horopter-0.4b-20261020; about 4.5 h on one RTX 4090, about 10 GB):

```bash
python training/train_joint.py --name prod_e400m_b2 \
  --arm hard --base jhu-clsp/ettin-encoder-400m --lr 1e-5 --typed_stance 1 --stance_undet 1 --ev_weight 0.5 \
  --prod $HOROPTER_DATA/data/v3/prod2 --max_state 2048 --max_tokens 16384 --max_state_slots 49152 --mm_group 1 --mm_skip_generic 1 \
  --long_eval data/v3/prod/eval_a25_test.jsonl \
  --drop_premises $HOROPTER_DATA/data/v3/e15/snli_heldout_image_premises.json \
  --drop_rows $HOROPTER_DATA/data/v3/e15/minhash_drop_narrow.json \
  --mix 74894:1722:4926:5358 --total_steps 86900 --save_every 10000 --keep_ckpt 1
```

**1B** (`prod_j1b_b2` = horopter-1b-20261020; about 10–11 h at about 0.44 s/step on a 4090, about 21.7 GB). The
arguments are the same apart from `--base`:

```bash
python training/train_joint.py --name prod_j1b_b2 \
  --arm hard --base jhu-clsp/ettin-encoder-1b --lr 1e-5 --typed_stance 1 --stance_undet 1 --ev_weight 0.5 \
  --prod $HOROPTER_DATA/data/v3/prod2 --max_state 2048 --max_tokens 16384 --max_state_slots 49152 --mm_group 1 --mm_skip_generic 1 \
  --long_eval data/v3/prod/eval_a25_test.jsonl \
  --drop_premises $HOROPTER_DATA/data/v3/e15/snli_heldout_image_premises.json \
  --drop_rows $HOROPTER_DATA/data/v3/e15/minhash_drop_narrow.json \
  --mix 74894:1722:4926:5358 --total_steps 86900 --save_every 10000 --keep_ckpt 1
```

Defaults that the recipe relies on:
- `--roles data/v3/roles/train_roles.jsonl` and `--conflict data/v3/e07/train_conflict.jsonl` (`--conflict_frac 1`);
- `--roles_bs 8`, `--conflict_bs 32`, `--evidence_bs 4`;
- `--max_blocks 96`, `--max_q 128`;
- `--head_lr 3e-4`, `--loc_weight 0.5`, `--mismatch 0.15`, `--w_rel 1`, `--w_nli 1`;
- `--warmup 0.03`, `--seed 13`.

The output is `$HOROPTER_OUTPUTS/e12/<name>/model.pt`: bf16 weights, plus the answer head.

How the training works:
- **Interruptions.** If a run is interrupted, re-run the same command with
  `--resume $HOROPTER_OUTPUTS/e12/<name>/ckpt/stepNNNNNN.pt`.
- **What `--long_eval` and the end evals need.**
  - `--long_eval` scores the held-out a25 test split at the end of training.
  - The end evals also use `--eval_pool` (`data/v3/e09/pool_eval.jsonl`) and `--roles_test`
    (`data/v3/roles/esnli_roles_test.jsonl`). Both are skipped when the file is missing.
  - `--eval_teacher` and `--long_eval_teacher` are optional teacher answer files, used for the Jev cross-entropy
    column.
- **Dry run.** `--dry_run 1` runs on the CPU and needs no model. It builds every stream, plans one epoch and reports
  the counts.

### 5. Export, adapters, abstention head

```bash
OUT=$HOROPTER_OUTPUTS
python scripts/export_hf.py --ckpt $OUT/e12/prod_e400m_b2/model.pt --base jhu-clsp/ettin-encoder-400m --max_state 2048 \
    --name horopter-0.4b --run prod_e400m_b2 --out $OUT/hf/horopter-0.4B
```

The research beta trained these adapters on the trunk (`--trunk` = the exported dir; see each script's `--help` for
the output and data paths). They are question-path LoRA, r 16, lr 3e-4, head lr 3e-4.

| adapter | data | steps | other |
|---|---|---|---|
| router | `data/v3/e24/router_train.jsonl` (training/build_adapter_data.py) | 1500 | |
| typed | `data/v3/e24/expert_typed.jsonl` | 600 | `--dev_n 150` |
| nli, relation, classification, knowledge, clause | `data/v3/e24/expert_<family>.jsonl` (from the E20 family files, `--cap 20000 --per_family 4000`) | 2500 each | |
| check | `data/v3/e38/check_train.jsonl` | 1500 | |
| check2 | `data/v3/e38/check_train_v2.jsonl` | 2500 | `--max_state 2048` |
| evidence (served) | train_evidence.py `--targets mixed --n_conflict 12000 --ref_weight 2 --steps 4000 --n_policy 6000` (e-SNLI roles 24k, E22 referent pairs 8k, e-SNLI conflict 12k, generated policies seed 2 with robot_sop held out) | 4000 | the research `ev_policy` run, served as `evidence` |

The family and check adapters were trained at the default `--max_state 512`, and check2 at 2048.

The abstention head (E35 `lr_T2`) is a logistic regression on 21 answer, relation and evidence scalars:
- **What it is fitted on.** The trunk's own errors, on `pool_eval` plus `data/v3/prod/eval_a25_dev.jsonl`.
- **How the features are extracted.** At `--max_state 2048` and `--max_blocks 192`, with the prod token budgets
  (16384 tokens, 49152 slots).
- **How it is checked.** On `eval_a25_test`, which is never fitted on.
- **Refit per trunk.** The beta head was fitted to prod_j1b, so each new trunk needs its own fit.

```bash
python training/fit_abstain_head.py features --trunk $OUT/hf/horopter-0.4B --out $OUT/abstain_feats \
    --set pool_eval=$HOROPTER_DATA/data/v3/e09/pool_eval.jsonl \
    --set a25_dev=$HOROPTER_DATA/data/v3/prod/eval_a25_dev.jsonl \
    --set a25_test=$HOROPTER_DATA/data/v3/prod/eval_a25_test.jsonl      # --max_state 2048 --max_blocks 192 (defaults)
python training/fit_abstain_head.py fit --feats $OUT/abstain_feats --out $OUT/abstain_head.json   # fit pool_eval+a25_dev, check a25_test
python scripts/export_hf.py ... --adapters_dir <adapters parent> --abstain_head <abstain_head.json> --out $OUT/hf/horopter-0.4B
```

### 6. Evaluation

```bash
python -m horopter.server --model $OUT/hf/horopter-0.4B --port 8420     # see horopter/server.py --help
export HOROPTER_URL=http://127.0.0.1:8420/v1/systemone
python eval/e40_build_sets.py --datasets /path/to/datasets             # ShARC / ConditionalQA / ProofWriter dev
python eval/e40_run_eval.py --models horopter,horopter-base,horopter-check && python eval/e40_score.py
python eval/e37_score.py --bench /path/to/e37/bench.jsonl --raw results/raw   # raw answers collected with eval/client.py
python eval/e37_family_breakdown.py --bench /path/to/e37/bench.jsonl --raw results/raw
```

The beta-2 gate (`gate_then_1b.sh`) compared prod_e400m_b2 with prod_e400m on the E37 bench, trunk only. The 1B run
went ahead only if the result was clean:
- AggreFact balanced accuracy and E0 macro-F1 each within 1 pt;
- JF100 and JevBench hard each down by no more than 2 pts, and at least one of them up.

## The mix (from `prod2_manifest.json`)

### Streams and steps

`--mix typed:roles:conflict:evidence = 74894:1722:4926:5358` over `--total_steps 86900`, i.e. 86.2% / 2.0% / 5.7% /
6.2% of optimiser steps. Each step is one batch of one stream, interleaved at random. The streams cycle as needed.

| stream | content | batch |
|---|---|---|
| typed | typed decision packs: per question one question block + one block per option. Plus 15% mismatched packs: questions over another row's state, trained as unrelated, donors from another document group (`--mm_group 1`), generic questions never mismatched (`--mm_skip_generic 1`) | token-budget batches: 16384 padded tokens, 49152 state slots, at most 96 blocks |
| roles | e-SNLI evidence-role packs (SUPPORT / CONFLICT on the state and the option side, UNSUPPORTED on the option side, cross-label negatives) | 8 packs |
| conflict | 183k e-SNLI contradiction pairs with premise highlights; refute state channel only, no stance loss | 32 packs |
| evidence | long-document gold evidence (a25: 21,432 rows; plus 3,353 new beta-2 entries for policy deciding rules and counterfactual changed words: 24,785 rows) | 4 packs |

The step counts are the beta's (prod_j1b):
- typed 1 epoch, roles 2 epochs, conflict 1 epoch, long evidence 1 epoch;
- beta 2 kept them unchanged. The e39 pool adds 42,428 typed packs (with repeats), so beta 2's typed stream covers a
  little less than one epoch of prod2.
- Recompute with a `--dry_run` (no `--mix`) if you change the selection.

Loss:
- typed answer CE + typed stance CE (support t, with `--stance_undet 1`: refute (1 − t)·d and undetermined
  (1 − t)·(1 − d), d = max_k t_k; precomputed `st` rows) + `--w_rel` × relevance BCE;
- roles / conflict / evidence: stance CE + `--ev_weight 0.5` × evidence channels + `--loc_weight 0.5` × localisation.

### Pools

All of these are paths under `$HOROPTER_DATA`, and none are shipped.

| pool | file | target rows |
|---|---|---|
| train | `data/v3/e09/pool_train.jsonl` (E09 pool: the decider-2b task mixture + typed_decisions) | 134,648 |
| a23 | `data/v3/e09/pool_a23.jsonl` (E14 gap pool) | 508,748 |
| a23anli | `data/v3/e09/pool_a23anli.jsonl` (ANLI) | 38,990 |
| a25 | `data/v3/e33/pool_a25.jsonl` (E33 long-document pool; the held-out a25 eval split removed) | 244,722 |
| e39 | `data/v3/e39/pool_e39.jsonl` (beta 2: written by build_prod2.py) | 14,012 rows: policy 12,000, check2 1,620, cf 392 |

Exclusions, applied first: the SNLI image-level split (train 298, a23 6,974 rows) and MinHash near-duplicates (57
rows).

### Targets

There are three teachers: Jev, decider (decider-2b) and clef-flash.
- **Calibration.** Each teacher gets one temperature per question type, fitted by grid NLL on a 10% fit bucket:

  | | noul | choice | score |
  |---|---|---|---|
  | jev | 1.044 | 2.378 | 2.000 |
  | decider | 1.756 | 1.242 | 0.805 |
  | clefflash | 1.091 | 1.354 | 0.738 |

- **Stack (`target_arm: guard`).** The calibrated teachers are combined in a linear pool with a pool temperature,
  fitted per cluster and shrunk toward the cluster's parent. On sources that decider was trained on, its share is
  capped at 1/n. For example, the `all` cluster has weights Jev 0.347, decider 0.333, clef-flash 0.320 and T 0.700.
  The per-cluster weights are in `stack` in the manifest.
- **Target.** With gold: 0.5 gold + 0.5 stack. Without gold: the stack. With gold but no teacher: the expected mixc
  prior.

  | source of the target | questions |
  |---|---|
  | gold + all three teachers | 1,431,249 |
  | gold + decider + clef-flash | 92,592 |
  | gold + Jev + clef-flash | 70 |
  | no gold, all three teachers | 243,947 |

- **Per-question fields.** Each question also carries:
  - the stance rows `st`;
  - the abstention target `ab`;
  - `w = 0` for questions not decidable inside the 2,048-token window (never trained);
  - `gq` for generic questions.
- **e39 rows.** These are exact labels smoothed to 0.97, with stance [t, 1 − t − 0.01, 0.01] and abstention 0.03.

### Selection: caps and repeats (`caps`)

- **Per-source decision caps.** The default cap is 20,000 decisions. Whole rows are kept: most teachers first, then a
  fixed hash order.
- **Sources with no cap:** typed_decisions, nli_anli, nli_scitail, nli_sick, evidence_inference_doc, race_article,
  ragtruth_multi, qasper_doc, control_nli, multirc_doc, hover, boolq_doc, attribution_long, contractnli_doc, wice_doc,
  folio.
- **Sources with their own caps:**

  | cap | sources |
  |---|---|
  | 60k | nli_mnli, msmarco_multi, squad_doc |
  | 40k | nli_snli, nli_vitaminc, nli_fever, nli_wanli |
  | 8k | shp_long, ultrafeedback_long, hh_rlhf_long, helpsteer2_long, helpsteer3_long |

- **Repeats (`k`).** typed_decisions ×4. e39: policy ×3, check2 ×3, cf ×4.
- **Totals.** The beta selection keeps 1,106,156 decisions in 646,549 rows across 96 sources. By cluster:

  | cluster | decisions |
  |---|---|
  | nli | 402k |
  | classification | 328k |
  | long_reading | 115k |
  | knowledge | 68k |
  | long_retrieval | 60k |
  | judge | 41k |
  | long_judge | 40k |
  | grounding | 16k |
  | retrieval | 15k |
  | long_nli | 15k |
  | typed | 6k |

- **e39 additions** (decisions, repeats included):

  | stream | decisions |
  |---|---|
  | policy | 169,110 |
  | check2 | 31,815 |
  | cf | 1,568 |

### State window

The state window is 2,048 tokens (`window`, `--max_state 2048`). Questions are capped at 128 tokens. No e39 state
exceeds the window.

## What is not shipped

All of the following are inputs that you must supply under `$HOROPTER_DATA`. None of them are in this repo.

**Typed pools and teacher answers**
- `data/v3/e09/pool_{train,a23,a23anli,eval}.jsonl`
- `data/v3/e09/raw/<teacher>.<pool>.jsonl`, the cached Jev, decider and clef-flash answers
- `data/v3/e33/pool_a25.jsonl`

**Derived data**
- `data/v3/prod/` and `data/v3/prod2/`, which steps 1 and 3 write
- `data/v3/e39/pool_e39.jsonl`
- `data/v3/e38/*`: generated policies, checks and counterfactual pairs, plus the LLM job and output files

**Evidence streams**
- `data/v3/roles/train_roles.jsonl` and `esnli_roles_test.jsonl` (e-SNLI evidence roles)
- `data/v3/e07/train_conflict.jsonl` (e-SNLI contradiction pairs)

**Exclusion lists**
- `data/v3/e15/snli_heldout_image_premises.json`
- `data/v3/e15/minhash_drop_narrow.json`

**Adapter data**
- `data/v3/e24/source_family.json`
- `data/v3/e24/expert_*.jsonl` and `router_*.jsonl`
- `data/v3/e20/train/*` (the family training files)
- the E22 referent pairs (evidence adapter)

**Eval-only.** These must never be trained on and are never copied into the repo:
- the E37 bench (JevBench, JF100, LLM-AggreFact, E0 test);
- `data/v3/prod/eval_a25_{dev,test}.jsonl`;
- `data/v3/e38/policy_v0.jsonl`;
- the E40 sets (ShARC, ConditionalQA, ProofWriter dev).
