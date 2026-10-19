# Porting notes: research code -> release

This repository is a curated port of the research code behind Horopter (internal name "mbnli-3"). It is a port, not
a rewrite: the ops, dtypes, packing and RNG order are the research code's. The port removed research-only paths,
experiment arms the beta-2 recipe does not use, and hard-coded paths. This file records:
- the file map;
- what was dropped;
- the exported config;
- how parity was checked;
- how to run the training recipe.

## File map

The research paths are relative to the research repository root.

### Package `horopter/`

| release | research source | notes |
|---|---|---|
| `horopter/model.py` | `src/v3/interaction.py` (InteractionModel, `masks`), `experiments/v3/e05_relation_roles/train_roles.py` (RoleModel, `ARMS["R3x"]`), `experiments/v3/e12_joint/train_joint.py` (`Answer`, `unrel_logit`), `src/v3/lora.py` | Merged into one `HoropterModel`, design D (= "design B": question-blind state) with all layers interacting, a MaxSim evidence head, the region head and the R3x channels. Parameter names are unchanged, so research checkpoints load as is. |
| `horopter/packing.py` | `train_joint.py` (`typed_pack`, `n_blocks`, `batches`, `_token_batches`, `pack_len`, `state_len`), `experiments/v3/e09_jev_distill/train_student.py` (`FastCollator`, `option_texts`), `experiments/v3/e08_arc2_pilot/analyze.py` (`options`), `interaction.py` (`Collator`, `to`), `train_roles.py` (`RoleCollator`, `word_map`), `train_joint.py` (`LongRoleCollator`) | The batching globals (`MAX_TOKENS`, `MAX_SLOTS`, `MAX_STATE`) are now arguments. |
| `horopter/losses.py` | `train_joint.py` (`run_typed`, `typed_loss`, `typed_stance_loss`), `train_roles.py` (`loss_fn`, `masked_bce`) | `STANCE_UNDET` is now an argument. |
| `horopter/runtime.py` | `experiments/v3/e24_dryrun_router/student.py` (`Student`, `items_of`, `target_of`) | `TRUNKS` and `find_ckpt` are replaced by `load(dir)` (the HF layout) and `load_checkpoint(model.pt, base)`. The adapter bank lives on the Student. |
| `horopter/evidence.py` | `student.py` (`Student.spans`, `Student.state_offsets`) | unchanged |
| `horopter/abstain.py` | `student.py` (`Student.abstain_features`), `serve.py` (`abstain_of` and the placeholder) | `AbstainHead` reads `abstain_head.json`. |
| `horopter/families.py` | `experiments/v3/e24_dryrun_router/families.py` | Keeps `FAMILIES`, `ROUTER_Q` (option order unchanged) and `router_state`. |
| `horopter/server.py` | `experiments/v3/e24_dryrun_router/serve.py` | Same wire format; see "Served names" below. |

### `training/`

| release | research source | notes |
|---|---|---|
| `training/train_joint.py` | `experiments/v3/e12_joint/train_joint.py` | `--prod` path only; flags of the beta-2 recipe kept |
| `training/joint_eval.py` | `train_joint.py` (`predict`, `roles_val`, `nll_of`), `train_student.py` (`summarise`, `items_of`), `analyze.py` (`probs_of`, `gold_index`, `apply_T`), `experiments/v3/e02_interaction_depth/train.py` (`auroc`, `avg_prec`), `train_roles.py` (`predict_roles`, `score_role`, `role_eval`), `src/v3/evalkit/metrics.py` (`ece`) | end-of-training evals |
| `training/train_adapter.py` | `experiments/v3/e24_dryrun_router/train_adapter.py` | `--trunk` is an exported dir |
| `training/train_evidence.py` | `experiments/v3/e24_dryrun_router/train_evidence.py` | data paths are arguments |
| `training/build_adapter_data.py` | `experiments/v3/e24_dryrun_router/build_data.py` | |
| `training/fit_abstain_head.py` | `experiments/v3/e35_abstention_head/features.py` + `export_head.py` (+ `Head` from `train_eval.py`) | subcommands `features` and `fit` |
| `training/data/*.py` | `experiments/v3/e38_effort_probes/{policy_gen,node_data,check_v2_data,cf_pairs,compile_real}.py`, `experiments/v3/e39_beta2_data/build_prod2.py`, `experiments/v3/e34_production_data/{build,lengths,common}.py` (+ `caps.json`) | data builders and generators |
| `training/tools/llm_fanout.py` | `experiments/v3/e21_xavier_compute/xq.py` | endpoints come only from a JSON file or URLs given as an argument |
| `training/recipes/beta2.md`, `prod2_manifest.json` | `e39_beta2_data/retrain_b2_400m.sh`, `gate_then_1b.sh`, `data/v3/prod2/manifest.json` | the recipe; the manifest is copied without data |

### Other directories

| release | research source |
|---|---|
| `eval/e37_score.py`, `eval/e37_family_breakdown.py` | `experiments/v3/e37_beta_bench/{score,family_breakdown}.py` |
| `eval/e40_build_sets.py`, `eval/e40_run_eval.py`, `eval/e40_score.py` | `experiments/v3/e40_rule_evals/{build_sets,run_eval,score}.py` |
| `eval/client.py`, `eval/narrow_reask.py` | `experiments/v3/e38_effort_probes/narrow_reask.py` (`ask()`) |
| `scripts/export_hf.py` | new: research checkpoint -> HF layout |
| `tests/test_parity.py` | new: release vs research, see "Parity" |

## What was dropped

**Model and trainer arms the recipe does not use** (`train_joint.py`):
- `--arm free` / `--arm soft` (the OptionHead answer and the residual). `Answer` now accepts only `hard`.
- `--score thresholds`: the cumulative threshold blocks, `thr_to_levels`, the monotonicity hinge (`--hinge`) and the
  threshold-violation report. Score questions use levels, as the recipe does.
- `--typed_rel_head`: the separate typed stance MLP (`trel`). No production checkpoint has one.
- `--lora` / `--lora_q`: training the trunk itself as a LoRA or a question-path LoRA. The recipe is a full fine-tune.
  LoRA is kept for adapters; `apply_lora` and `merge_lora` are in `horopter/model.py`.
- `--llrd` and `--sp_decay`: the E18 anti-forgetting arms.
- The non-`--prod` typed path (`--pools`, `--teachers`, and calibration of the raw teacher files at train time). The
  production data carries its targets precomputed, so `--prod` is now required.
- The A1.1 NLI benchmark evals at the end of training (`eval_all` over `comp_test`, `anli_test`, `mnli_mm`, `e0_dev`
  and `esnli_test`, with their `--evals` flag). The E37 bench in `eval/` covers these.
- The dry-run's `.npz` batch-shape dump. The JSON report is kept and now goes to `<out_dir>/dryrun.json`.

**Model code paths that are never reached in design B with all layers interacting:**
- the `E`, `F`, `G`, `H` and `A` designs;
- the per-question fork tensors (`f_src`, `f_pos`, `f_blk`) in the collator;
- the `iso` and `full` mask modes, and `cross_global`, which is always False;
- the `bilinear` and `xattn` evidence heads;
- the T5Gemma-2 trunk.

The bilinear head's `ev_s` / `ev_q` linears are still created, unused, so that checkpoints load strictly. `RoleCollator`'s
rule channels (R6 arm) are also dropped.

**Runtime and server:**
- `find_ckpt` (the research output and pod-run search) and the `TRUNKS` table.
- The `--trunk` / `--adapters` defaults, which pointed into the research tree.

**Abstention:**
- the T1 (committee target) and T3 heads, and the MLP heads;
- the 5-fold cross-fit comparison (`train_eval.py`);
- the `pool_eval_mm` and `train_sample` feature sets, and the question-vector `h` features.

What remains is exactly the served head: `lr_T2`, a logistic head on 21 features, fitted on the trunk's own errors.

**Not ported at all:**
- `sdk_probe.py`, and every teacher-labelling client (`label.py` in E08 and E09) or other code that calls a paid API.
- `eval_e2e.py`, `eval_user_evidence.py` and `train_laya_adapter.py` (research evals and a teacher adapter).
- The E37 `bench.py`, `retry_truncated.py` and `policy_eval.py` drivers, and the E40 `decoder_baseline.py`.
- The research shell queues.

## Changes in behaviour (none in numerics)

- **Served names.**
  - The default model id prefix is `horopter` (`horopter`, `horopter-base`, `horopter-<adapter>`, with `-strict`).
  - Every name is also served under each `--alias` prefix. The default alias is `mbnli-3-beta`, so beta clients keep
    working; pass `--alias ''` to turn it off.
  - The research `mbnli-3-dryrun` prefix is gone.
- **Wire format.** It is unchanged.
  - The per-answer and per-response extension block is still called `mbnli`, so beta clients parse it unchanged.
  - Requests may send their options as `mbnli` or `horopter`.
  - Only the `version` string changed: `3.0.0+contract1 trunk=<name>:<run>`. The `release_date` comes from the
    exported config.
- **Device.** The research code read a module global (`DEV`) set when it was imported. The release takes a `device`
  argument, defaulting to CUDA if available. Autocast to bf16 is on exactly when the device is CUDA, as before.
- **Adapter trainers.** They build the encoder with `from_pretrained(base)` before loading the trunk weights
  (`runtime.load(..., pretrained_init=True)`). This replays the research construction, and with it the torch RNG
  stream that the LoRA A init draws from. Serving uses `from_config`, so it needs no base weights; the outputs are
  identical either way.
- **Defaults kept from the research code, even where they look odd.**
  - `train_adapter.py` and `train_evidence.py` default to `--max_state 512` even on 2048-token trunks; the beta's family
    and check adapters were trained that way.
  - The roles collator keeps the base collator's `max_state=256`, `max_q=64`.
- **Outputs.** Results go to `--out_dir` / `--out`, defaulting under `$HOROPTER_OUTPUTS` (default `outputs/`).
  Relative data paths resolve against `--data_root` (`$HOROPTER_DATA`, default `.`). The relative paths in the
  manifest (`data/v3/...`) are kept, so a copy of the research `data/` tree under `$HOROPTER_DATA` works unchanged.

## Exported layout and config fields

`scripts/export_hf.py` writes:

```
config.json             see below
model.safetensors       trunk + heads, keys as in training; "answer.tau" = the answer head; checkpoint dtypes (bf16 trunk)
tokenizer.json, tokenizer_config.json, special_tokens_map.json   the base model's tokenizer
adapters/<name>/adapter.safetensors + adapter.json   question-path LoRA A/B + all head tensors + answer.tau (fp32)
abstain_head.json       {"name", "features", "mu", "sd", "w", "b", "trunk", "max_state", "n_fit", ...}
```

`config.json`:

| field | meaning |
|---|---|
| `format` | `horopter-hf-1` (checked by `horopter.runtime.load`) |
| `architecture` | `HoropterModel` |
| `name`, `run`, `release_date` | model id, training run name, served release date |
| `base_model` | the Ettin id, e.g. `jhu-clsp/ettin-encoder-400m` (tokenizer fallback) |
| `base_config` | the full transformers config of the base, so the encoder is rebuilt with no download (`AutoConfig.for_model`) |
| `hidden_size`, `num_hidden_layers`, `local_attention` | copied from the base config, for reference |
| `max_state`, `max_q` | the state window (tokens incl. [CLS]/[SEP]) and the question-block cap the trunk was trained with |
| `weights_dtype` | the dtypes stored in `model.safetensors`; the runtime upcasts to fp32 as the research runtime did |
| `design` | design B: question-blind cacheable state, all layers interacting, `cross_global` false, sdpa |
| `heads` | relation labels, `evidence_head: maxsim`, `loc_head: true`, the R3x channels, the state channel names |
| `answer` | `arm: hard`, the type ids, `score_mode: levels` |
| `lora` | adapter rank `r` (16), `alpha` (2r), gated to question tokens, on every trunk Linear |
| `adapters` | names under `adapters/` |
| `abstain_head` | file name, or null (the server then uses the entropy placeholder) |

`horopter.runtime.load(path)` reads this layout. A hub repo id also works if `huggingface_hub` is installed. It
returns a `Student` with:
- `.bank`: every adapter, plus `base`;
- `.abstain_head`;
- `.answer()`, `.answer_one_each()` and `.use(name)`.

## Parity

All the checks below ran on CPU in fp32, with no GPU visible.

- **Serving (`tests/test_parity.py`).**
  - Setup: research run `prod_e400m` (400M, `--max_state 2048`), exported with the `check` adapter and the E35
    `lr_T2` head. The head was fitted to the 1B; here it only tests the plumbing.
  - Requests: 20 requests, 44 questions. They cover noul (with and without criteria), choice and score; text and JSON
    states; one to three questions per state; dict instructions; evidence `all`, `chosen` and `off`; the trunk alone
    and the `check` adapter.
  - Research side: `serve.build_app(trunk="prod400m")` and its `Student`, run in-process.
  - Release side: `horopter.runtime.load` and `horopter.server.build_app`.
  - Result: the max abs difference is **0.0** for probabilities, stance rows, p(unrelated), the 21 abstention
    features and the raw evidence channel scores. The served JSON is identical, including every evidence span (60)
    and `abstain.p`. The `mbnli-3-beta-*` alias returns the same answers.
- **Joint trainer.**
  - Setup: a mini production set (300 rows from the e39, a25 and train pools, with evidence packs), on Ettin-150m,
    8 steps covering all four streams, with the recipe's flags (`--max_tokens`, `--mm_group`, `--mm_skip_generic`,
    `--stance_undet`, the drop lists).
  - Run deterministically (`torch.use_deterministic_algorithms(True)`, one thread), research `train_joint.py` and
    `training/train_joint.py` give **bitwise-identical** weights and identical held-out predictions.
  - Multi-threaded CPU runs differ run-to-run in the last bf16 bit, and that happens for either codebase.
- **Adapter trainer.** On the same trunk, research `train_adapter.py` and `training/train_adapter.py` (6 steps) give
  bitwise-identical adapters.
- **Abstention features and fit.**
  - Setup: the research `features.py` (prod_e400m, 40 rows each of pool_eval, a25_dev and a25_test; 412 items) and
    `fit_abstain_head.py features`.
  - The features agree within 1.6e-5, and `wrong` / `long` are identical. Repeating the release run on pool_eval gave a
    max diff of 0.0, so the residual is multi-threaded CPU noise.
  - The research `export_head.py` fit and `fit_abstain_head.py fit` on these features agree within 1.2e-7 in
    mu / sd / w, b is equal, and the a25_test check numbers are identical.

Run the serving test with:

```bash
export HOROPTER_RESEARCH_ROOT=/path/to/research/checkout HOROPTER_PARITY_EXPORT=/tmp/hf_400m
python scripts/export_hf.py --ckpt $HOROPTER_RESEARCH_ROOT/outputs/v3/e12/prod_e400m/model.pt \
    --base jhu-clsp/ettin-encoder-400m --max_state 2048 \
    --adapter $HOROPTER_RESEARCH_ROOT/outputs/v3/e24/prod400m/check \
    --abstain_head $HOROPTER_RESEARCH_ROOT/outputs/v3/e35/abstain_lr_T2.json --out $HOROPTER_PARITY_EXPORT
CUDA_VISIBLE_DEVICES="" python -m pytest -s tests/test_parity.py      # skipped when the two paths are not set
```

## Running the training recipe

`training/recipes/beta2.md` has the full pipeline, with the exact 400M and 1B command lines. In short:
1. Build the beta-2 streams and `prod2` (`training/data/`).
2. Run `training/train_joint.py` with the recipe's flags.
3. Export the trunk with `scripts/export_hf.py`.
4. Train the adapters (`training/build_adapter_data.py`, `training/train_adapter.py`, `training/train_evidence.py`)
   on the exported trunk.
5. Fit the abstention head with `training/fit_abstain_head.py features` and then `fit`.
6. Export again, now with `--adapters_dir` and `--abstain_head`.
7. Serve with `python -m horopter.server --model <dir>`, then evaluate with `eval/`.

None of the data is in this repository. Every pool, target, length file, roles / conflict stream, exclusion list and
eval set has to be supplied under `$HOROPTER_DATA`; the recipe lists them. The eval sets (LLM-AggreFact, JevBench,
JF100, the a25 test split, policy_v0, the E40 sets) are eval-only: they are never trained on and never copied here.
