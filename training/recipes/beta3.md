# Beta-3 training recipe (horopter-0.4b-20261104, horopter-1b-20261104)

Beta-3 is the [beta-2 recipe](beta2.md) with one data change: the rule-application stream adds the outcomes that do not decide. In the research repo the runs were `prod_e400m_b3` (400M) and `prod_j1b_b3` (1B), experiment E47.

Beta-2 had a determined answer for every rule-application item. The trained models then stopped saying "irrelevant" or "more information needed" on real rule text (ShARC recall 75 → 36% and 16 → 0.5% at 1B).

## What changes against beta-2

**Data: `data/v3/prod3` in place of `prod2`.** It is the E34 production data (`data/v3/prod`, unchanged) plus pool `e47`, which holds four exact-label streams:

| stream | source | rows / decisions | change from beta-2 |
|---|---|---|---|
| policy3 | policy generator v3, seed 3, both registers (JSON and prose); robot_sop held out | 12,002 / 210,582 | replaces the v0 generator stream (see below) |
| sharc | ShARC train (CC BY-SA 3.0), in the E40 eval format: the 4-way question (yes / no / irrelevant / more_info) and, for yes/no items, the user's question as a yes/no; listed twice | 21,890 / 71,440 | new |
| check2 | single-condition check sets | 1,620 / 31,815 | unchanged |
| cf | counterfactual minimal pairs | 392 / 1,568 | unchanged |

Generator v3 keeps v0's rules, rendering and execution, and adds two outcomes:
- **more_info:** a case value that a check needs is missing (absent, or stated as "unknown"). Execution walks the checks in order, and reaching a check whose value is missing gives more_info. A missing value *after* the deciding check does not matter, and the item keeps its decided answer.
- **not_applicable:** the policy carries a scope clause ("This policy covers in-store purchases only."), and the case falls outside it.

Each outcome has three option phrasings. Only half of the policies state the missing-information fallback in words. Twins now also include supplying the missing value or moving the case into scope. Node questions add "does the case state the <field>?" and the scope question.

Of the 12,002 policy rows, 5,680 are decided, 3,082 end in more_info, 2,228 in a stated default and 1,012 in not_applicable.

**Decontamination.** Every new state is checked against the E37 bench (13-gram overlap) and against the policy_v0 and policy_v3 dev states. ShARC train and dev share no rule snippet or tree. ProofWriter and ConditionalQA stay out of training.

**Training.** Identical to beta-2: same flags, `--mix 74894:1722:4926:5358`, `--total_steps 86900`. Only `--prod .../prod3` changes.
- The 1B run also used `--skip_oom 1`. One oversized evidence batch did not fit in 24 GB, deterministically, so two evidence batches (steps 51,812 and 60,875 of 86,900) were dropped and logged.

**Adapters, router and abstention head.** These were retrained on the beta-3 trunks with the beta-2 commands.

**New in the bank: `extract`.** It is the field-extraction adapter used by `fields` requests:
- a loraq r16 adapter initialised from `evidence`, trained for 1,500 steps on generated typed-field documents;
- extra training rows for values with inner punctuation, and for "hard absent" fields (the asked role is missing, but a same-type value is present for another role);
- holdout types: IBAN, MAC address, postal code.

The 0.4B bank also has `router_q`, an optional router for `--route_layout question`.

## Not yet in this repo

The beta-3 data builders (`policy_gen3.py`, `build_prod3.py`), the `--skip_oom` flag of `train_joint.py` and the `extract` trainer are still in the research repo (experiments E47, E43, E56, E56b). They will be ported here the same way as the beta-2 builders ([PORTING.md](../../docs/PORTING.md)), with a byte-identity check of the rebuilt `prod3`.
