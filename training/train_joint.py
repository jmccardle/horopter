# SPDX-License-Identifier: MIT
"""Joint trainer: every Horopter head trained together on one design-B trunk (full fine-tune).

Streams, interleaved at random by step (shares fixed by --mix typed:roles:conflict:evidence):
  typed     production rows (--prod: manifest.json + targets/ + lengths/ per pool): precomputed soft targets (gold +
            stacked teacher committee), stance rows, window flags. Per question: one question block (its relation head
            gives relevance) plus one isolated block per option. A --mismatch share of extra packs re-asks a batch's
            questions over another row's state (half from the same source: near-miss negatives); only their question
            blocks are trained, as unrelated.
  roles     e-SNLI evidence-role packs (--roles): SUPPORT / CONFLICT on the state and on the option side, UNSUPPORTED
            on the option side, cross-label negatives. (Served names: support / refute / unsupported.)
  conflict  e-SNLI contradiction pairs with premise highlights (--conflict) -> the refute state channel only; their
            relation CE is removed (a single-label stream would teach stance "always refute").
  evidence  long-state evidence packs listed by the manifest's "evidence" file (gold spans on prod rows).
Answer (the hard arm): option logit = tau_type * log p(support) under the 3-way stance softmax.
Loss = typed CE + typed_stance * stance CE + w_rel * relevance BCE  |  w_nli * roles loss (stance CE + ev_weight *
channels + loc_weight * localisation).

    python training/train_joint.py --name prod_e400m_b2 --arm hard --prod data/v3/prod2 ... (training/recipes/beta2.md)
-> <out_dir>/{model.pt, eval_preds.jsonl, results.json}; <out_dir> = --out_dir or $HOROPTER_OUTPUTS/e12/<name>
   (default outputs/e12/<name>). Relative data paths resolve against --data_root ($HOROPTER_DATA, default ".").
"""
import argparse
import copy
import gzip
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from horopter.losses import roles_loss, run_typed, typed_loss, typed_stance_loss  # noqa: E402
from horopter.model import CHANS, Answer, HoropterModel  # noqa: E402
from horopter.packing import (FastCollator, LongRoleCollator, RoleCollator, batches, n_blocks, option_texts,  # noqa: E402
                              options, pack_len, state_len, to, typed_pack)
import joint_eval as JE  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"   # cpu: smoke tests only (CUDA_VISIBLE_DEVICES="")
MAX_TOKENS, MAX_SLOTS, MAX_STATE = 0, 0, 512           # set from the flags in main()
MM_GROUP = False     # --mm_group: mismatch donors from another document group
MM_GENERIC = False   # --mm_skip_generic: generic questions (asked of >= 20 states) are never mismatched


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0


def bcfg():
    return {"max_tokens": MAX_TOKENS, "max_slots": MAX_SLOTS, "max_state": MAX_STATE}


def mismatched_rows(rows, rng, n, by_src, max_len=None, batch=None):
    """n (other-state, items) pairs; half the donors come from the same source (near-miss negatives). Under
    --max_tokens the questions come from the batch's own rows (batch) and the donor state is no longer than the
    batch's longest state (max_len), so a mismatched pack cannot blow the padded-token budget."""
    out = []
    for _ in range(n):
        r, its = batch[rng.randrange(len(batch))] if batch else rows[rng.randrange(len(rows))]
        if MM_GENERIC:   # a generic question ("Which response is better?") is not unrelated to another state
            its = [it for it in its if not it.get("generic")]
            if not its:
                continue
        pool = by_src[r["source"]] if rng.random() < 0.5 and len(by_src[r["source"]]) > 1 else None
        for _ in range(5):
            d = pool[rng.randrange(len(pool))] if pool else rows[rng.randrange(len(rows))][0]
            if d["id"] != r["id"] and d["state"] != r["state"] and \
                    not (MM_GROUP and d.get("group") is not None and d.get("group") == r.get("group")) and \
                    (max_len is None or state_len(d) <= max_len):
                out.append((d["state"], its))
                break
    return out


def resolve(root, p):
    p = Path(p)
    return p if p.is_absolute() else Path(root) / p


def load_prod(prod, root, drop, kept, drows):
    """Production data (manifest.json, targets/<pool>.jsonl.gz, lengths/<pool>.json.gz): pool rows referenced by id,
    targets precomputed (stacked committee + gold, stance with the undetermined split, abstention, window flags).
    -> typed [(row, items)] with rows repeated k times, temperatures, evidence packs (gold spans).
    Pool and evidence paths in the manifest are relative to `root` (the data root)."""
    man = json.load(open(prod / "manifest.json"))
    evd = [json.loads(line) for line in gzip.open(resolve(root, man["evidence"]), "rt")]
    ev_ids = {d["id"] for d in evd}
    typed, n_dec, n_skip, ev_rows = [], 0, Counter(), {}
    for pool, path in man["pools"].items():
        tg = {}
        with gzip.open(prod / "targets" / f"{pool}.jsonl.gz", "rt") as f:
            for line in f:
                d = json.loads(line)
                tg[d["id"]] = d
        ln = json.load(gzip.open(prod / "lengths" / f"{pool}.json.gz", "rt"))
        ln = {i: (s, q) for i, s, q in zip(ln["ids"], ln["s"], ln["q"])}
        for line in open(resolve(root, path)):
            r = json.loads(line)
            d = tg.get(r["id"])
            if d is None:
                continue
            if r["id"] in ev_ids:
                ev_rows[r["id"]] = r                        # the evidence packs need these states
            if not d.get("k") or r["source"] in drop or not kept(r["state"]) or (f"pool_{pool}", r["id"]) in drows:
                continue
            r["group"] = d.get("g") or "s|" + str(hash(r["state"].strip()))
            its = []
            for qid, q in r["questions"].items():
                t = d["q"].get(qid)
                if t is None or not t.get("w", 1):         # w = 0: not decidable within the state window
                    n_skip["no_target_or_outside_window"] += 1
                    continue
                if len(t["t"]) != len(options(q)):
                    n_skip["option_mismatch"] += 1
                    continue
                its.append({"row": r["id"], "qid": qid, "q": q, "type": q["type"], "src": r["source"],
                            "heldout": r.get("heldout"), "gold": None, "t": {}, "target": np.asarray(t["t"]),
                            "stance": t.get("st"), "abst": t.get("ab"), "generic": bool(t.get("gq"))})
            if not its:
                continue
            s_tok, q_tok = ln.get(r["id"], (len(r["state"]) // 3, 0))
            # question tokens of the kept items only (blocks of dropped questions are not packed)
            frac = sum(1 + len(options(it["q"])) for it in its) / sum(1 + len(options(q)) for q in r["questions"].values())
            r["_len"] = (s_tok, int(q_tok * frac))
            typed += [(r, its)] * int(d["k"])
            n_dec += len(its) * int(d["k"])
    ev = []
    for d in evd:
        r = ev_rows.get(d["id"])
        if r is None or not kept(r["state"]):
            continue
        qs = []
        for x in d["questions"]:
            q = r["questions"][x["qid"]]
            instr = q.get("instructions") or x["qid"]
            qs += [{"text": f"{instr} Answer: {o}", "label": lab, "words": x["words"], "src": "a25ev",
                    "id": f"{d['id']}#{x['qid']}"} for o, lab in zip(option_texts(q), x["lab"]) if lab >= 0]
        if qs:
            ev.append({"state": r["state"], "n_units": 1, "questions": qs, "source": d["source"]})
    print(f"[prod] {len(typed)} typed packs ({n_dec} decisions, repeats included), {len(ev)} evidence packs; "
          f"skipped {dict(n_skip)}; window {man['window']}", flush=True)
    return typed, man["temperatures"], ev


def dry_run(a, typed, roles, conf, ev, col_t, col_r, col_e, ns, by_src, out_dir):
    """Every stream loaded as training would load it; one epoch of typed and evidence batches planned with the run's
    batching flags; a sample of the longest batches collated; counts and lengths -> <out_dir>/dryrun.json. No model."""
    t0 = time.time()
    W, tok = a.max_state, col_t.tok

    def pct(x):
        x = np.asarray(x)
        return {"n": int(len(x)), "mean": round(float(x.mean()), 1), **{f"p{q}": int(np.percentile(x, q))
                                                                        for q in (50, 90, 99)}, "max": int(x.max())}
    rep = {"config": vars(a), "steps": dict(zip(("typed", "roles", "conflict", "evidence"), ns))}
    bl = list(batches(typed, a.max_blocks, random.Random(0), **bcfg()))
    rng = random.Random(a.seed)
    st_len = [r["_len"][0] if "_len" in r else len(tok(r["state"], add_special_tokens=False)["input_ids"])
              for r, _ in typed]
    slen = {r["state"]: s for (r, _), s in zip(typed, st_len)}
    B, T, tokens, padded, slots, blocks = [], [], 0, 0, [], []
    for b in bl:
        mm = mismatched_rows(typed, rng, round(len(b) * a.mismatch), by_src,
                             *((max(state_len(r) for r, _ in b), b) if MAX_TOKENS else ()))
        lens = [pack_len(r, its, MAX_STATE) for r, its in b]
        for s, its in mm:
            sl = slen.get(s, len(s) // 4)
            lens.append(min(sl, W - 2) + 2 + sum(len(it["q"].get("instructions") or "") // 4 + 2 for it in its))
        B.append(len(lens))
        T.append(max(lens))
        tokens += sum(lens)
        padded += len(lens) * max(lens)
        nb = sum(n_blocks(its) for _, its in b) + sum(len(its) for _, its in mm)
        blocks.append(nb)
        slots.append(nb * min(max((r["_len"][0] if "_len" in r else 0) for r, _ in b) + 2, W))
    src_dec, src_rows, src_tok = Counter(), Counter(), Counter()
    for r, its in typed:
        src_dec[r["source"]] += len(its)
        src_rows[r["source"]] += 1
        src_tok[r["source"]] += pack_len(r, its, MAX_STATE)
    rep["typed"] = {
        "packs": len(typed), "decisions": sum(src_dec.values()), "batches_per_epoch": len(bl),
        "state_tokens": pct(st_len), "states_over_window": int(sum(s > W - 2 for s in st_len)),
        "batch_rows": pct(B), "batch_T": pct(T), "batch_blocks": pct(blocks), "batch_state_slots": pct(slots),
        "tokens_per_epoch": int(tokens), "padded_tokens_per_epoch": int(padded),
        "by_source": {s: {"rows": src_rows[s], "decisions": src_dec[s], "pack_tokens": src_tok[s]} for s in sorted(src_dec)},
    }
    ev_len = [len(x) for x in tok([p["state"] for p in ev], add_special_tokens=False)["input_ids"]] if ev else []
    if ev:
        eb = [ev[i:i + a.evidence_bs] for i in range(0, len(ev), a.evidence_bs)]
        rep["evidence"] = {"packs": len(ev), "questions": sum(len(p["questions"]) for p in ev),
                           "by_source": dict(Counter(p["source"] for p in ev)),
                           "state_tokens": pct(ev_len), "batches_per_epoch": len(eb)}
    rep["roles"] = {"packs": len(roles)}
    rep["conflict"] = {"packs": len(conf)}
    shapes = []
    order = sorted(range(len(bl)), key=lambda i: -T[i])[:3] + random.Random(1).sample(range(len(bl)), min(3, len(bl)))
    for i in order:
        pg = [typed_pack(r["state"], its) for r, its in bl[i]]
        bt = col_t([p for p, _ in pg])
        assert bt["s_tok"].shape[1] <= W, bt["s_tok"].shape
        shapes.append({"stream": "typed", "ids": list(bt["ids"].shape), "s_tok": list(bt["s_tok"].shape)})
    if ev:
        for i in random.Random(2).sample(range(len(eb)), min(2, len(eb))):
            be = col_e(eb[i])
            shapes.append({"stream": "evidence", "ids": list(be["ids"].shape), "positive": int(be["es_t"].sum())})
    if roles:
        shapes.append({"stream": "roles", "ids": list(col_r(roles[:a.roles_bs])["ids"].shape)})
    rep["collated_samples"] = shapes
    rep["sec"] = round(time.time() - t0)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out_dir / "dryrun.json", "w"), indent=1)
    print(f"[{a.name}] dry run -> {out_dir / 'dryrun.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", default="hard", choices=["hard"], help="answer arm (the release keeps the hard bottleneck)")
    ap.add_argument("--base", default="jhu-clsp/ettin-encoder-400m")
    ap.add_argument("--prod", required=True, help="production data dir (manifest.json, targets/, lengths/)")
    ap.add_argument("--data_root", default=os.environ.get("HOROPTER_DATA", "."),
                    help="root that relative data paths (manifest pools, --roles, --conflict, evals) resolve against")
    ap.add_argument("--out_dir", default=None, help="default $HOROPTER_OUTPUTS/e12/<name> (outputs/e12/<name>)")
    ap.add_argument("--roles", default="data/v3/roles/train_roles.jsonl", help="e-SNLI evidence-role packs")
    ap.add_argument("--conflict", default="data/v3/e07/train_conflict.jsonl", help="e-SNLI contradiction pairs")
    ap.add_argument("--max_blocks", type=int, default=96, help="typed blocks per batch (question + option blocks)")
    ap.add_argument("--max_state", type=int, default=512)
    ap.add_argument("--max_q", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--typed_epochs", type=float, default=1.0)
    ap.add_argument("--roles_epochs", type=float, default=1.0)
    ap.add_argument("--roles_bs", type=int, default=8)
    ap.add_argument("--conflict_frac", type=float, default=1.0, help="share of the e-SNLI contradiction pairs")
    ap.add_argument("--conflict_bs", type=int, default=32)
    ap.add_argument("--mismatch", type=float, default=0.15, help="mismatched packs per typed row")
    ap.add_argument("--w_rel", type=float, default=1.0, help="relevance loss weight (typed stream)")
    ap.add_argument("--w_nli", type=float, default=1.0, help="weight of the roles, conflict and evidence streams")
    ap.add_argument("--ev_weight", type=float, default=2.0)
    ap.add_argument("--loc_weight", type=float, default=0.5)
    ap.add_argument("--typed_stance", type=float, default=0.0,
                    help="weight of stance supervision on typed option blocks (support t_k, refute 1 - t_k)")
    ap.add_argument("--stance_undet", type=int, default=0,
                    help="typed stance targets: the precomputed rows, else 1 - t_k split between refute and "
                         "undetermined by determinacy max_k t_k")
    ap.add_argument("--limit", type=int, default=None, help="typed train rows (smoke tests)")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--drop_sources", default=None, help="JSON list of typed sources to leave out")
    ap.add_argument("--mix", default=None, help="explicit step shares typed:roles:conflict[:evidence]; default = "
                                                "proportional to stream size")
    ap.add_argument("--total_steps", type=int, default=None, help="with --mix: total optimiser steps")
    ap.add_argument("--warmup", type=float, default=0.03, help="linear warm-up share of steps")
    ap.add_argument("--drop_premises", default=None, help="JSON list of premises to leave out of every stream")
    ap.add_argument("--drop_rows", default=None, help="JSON list of [stream, row id] to leave out")
    ap.add_argument("--evidence_bs", type=int, default=4, help="evidence packs per step (4th --mix share)")
    ap.add_argument("--max_tokens", type=int, default=0, help="typed batches: padded-token budget B x T with length "
                                                              "bucketing (0 = block-count batching)")
    ap.add_argument("--max_state_slots", type=int, default=0, help="typed batches: blocks x state length budget (0 = off)")
    ap.add_argument("--mm_group", type=int, default=0, help="mismatch donors from another document group")
    ap.add_argument("--mm_skip_generic", type=int, default=0, help="generic questions (gq flag) are never mismatched")
    ap.add_argument("--eval_pool", default="data/v3/e09/pool_eval.jsonl", help="held-out typed rows with gold "
                                                                               "(skipped if missing)")
    ap.add_argument("--eval_teacher", default=None, help="teacher answers on --eval_pool (JSONL id/answers) for ce_jev")
    ap.add_argument("--long_eval", default=None, help="held-out long-state eval files (comma list): accuracy by "
                                                      "source, long vs short state, decidable within the window")
    ap.add_argument("--long_eval_teacher", default=None, help="teacher answers on --long_eval")
    ap.add_argument("--roles_test", default="data/v3/roles/esnli_roles_test.jsonl", help="e-SNLI roles test "
                                                                                         "(skipped if missing)")
    ap.add_argument("--eval_only", default=None, help="path to a model.pt (or stepNNNNNN.pt): skip training, score it")
    ap.add_argument("--save_every", type=int, default=0, help="resumable checkpoint every N steps (0 = final model only)")
    ap.add_argument("--keep_ckpt", type=int, default=1, help="mid-run checkpoints kept (the newest N)")
    ap.add_argument("--save_opt", type=int, default=1, help="include the optimiser state (exact resume)")
    ap.add_argument("--ckpt_dir", default=None, help="mid-run checkpoint dir (default <out_dir>/ckpt)")
    ap.add_argument("--resume", default=None, help="a --save_every checkpoint: continue the run from its step (same flags)")
    ap.add_argument("--dry_run", type=int, default=0, help="CPU: build every stream and one epoch of batches, collate "
                                                           "a sample, report counts and lengths, exit (no model)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    global MAX_TOKENS, MAX_SLOTS, MAX_STATE, MM_GROUP, MM_GENERIC
    MAX_TOKENS, MAX_SLOTS, MAX_STATE = a.max_tokens, a.max_state_slots, a.max_state
    MM_GROUP, MM_GENERIC = bool(a.mm_group), bool(a.mm_skip_generic)
    root = Path(a.data_root)
    out_dir = Path(a.out_dir) if a.out_dir else Path(os.environ.get("HOROPTER_OUTPUTS", "outputs")) / "e12" / a.name
    rng = random.Random(a.seed)

    drop = set(json.load(open(resolve(root, a.drop_sources)))) if a.drop_sources else set()
    dprem = set(json.load(open(resolve(root, a.drop_premises)))) if a.drop_premises else set()
    drows = {tuple(x) for x in json.load(open(resolve(root, a.drop_rows)))} if a.drop_rows else set()

    def kept(state):   # the premise is the state (nli_snli, roles, conflict) or the "Text 1:" line (snli)
        s = state.strip()
        return not dprem or (s not in dprem and s.split("\n")[0].removeprefix("Text 1: ").strip() not in dprem)
    typed, Ts, ev_packs = load_prod(resolve(root, a.prod), root, drop, kept, drows)
    typed = random.Random(a.seed).sample(typed, a.limit) if a.limit else typed   # smoke: a random sample
    by_src = defaultdict(list)
    for r, _ in typed:
        by_src[r["source"]].append(r)
    ep = resolve(root, a.eval_pool) if a.eval_pool else None
    evset = JE.eval_set([json.loads(l) for l in open(ep)][: a.eval_limit], a.eval_teacher, Ts) \
        if ep and ep.exists() else []

    roles = [r for i, r in enumerate(map(json.loads, open(resolve(root, a.roles))))
             if kept(r["state"]) and ("roles", f"roles:{i}") not in drows]
    conf = [r for r in map(json.loads, open(resolve(root, a.conflict)))
            if kept(r["premise"]) and ("conflict", r["id"]) not in drows]
    if dprem or drows:
        print(f"[{a.name}] --drop_premises {len(dprem)} / --drop_rows {len(drows)}: roles {len(roles)}, conflict {len(conf)} kept", flush=True)
    conf = random.Random(a.seed).sample(conf, int(len(conf) * a.conflict_frac))
    conf = [{"state": r["premise"], "n_units": 1,
             "questions": [{"text": r["hyp"], "label": 1, "words": r["gold"], "src": "esnli", "id": r["id"]}]} for r in conf]

    tok = AutoTokenizer.from_pretrained(a.base)
    col_t = FastCollator(tok, max_state=a.max_state, max_q=a.max_q)
    col_r = RoleCollator(tok, CHANS, cross_neg=True)
    col_e = LongRoleCollator(tok, CHANS, cross_neg=True, max_state=a.max_state, max_q=a.max_q)

    n_t = math.ceil(sum(1 for _ in batches(typed, a.max_blocks, random.Random(0), **bcfg())) * a.typed_epochs)
    n_r = math.ceil(len(roles) / a.roles_bs * a.roles_epochs)
    n_c = math.ceil(len(conf) / a.conflict_bs)
    n_e = math.ceil(len(ev_packs) / a.evidence_bs)
    if a.mix:   # fixed stream proportions, independent of stream sizes (streams cycle as needed)
        fr = [float(x) for x in a.mix.split(":")]
        fr += [0.0] * (4 - len(fr))           # optional 4th share = the evidence stream
        if fr[3] and not ev_packs:
            raise SystemExit("--mix has an evidence share but there are no evidence packs")
        total = a.total_steps or (n_t + n_r + n_c + n_e)
        n_t, n_r, n_c, n_e = (round(total * f / sum(fr)) for f in fr)
    kinds = ["t"] * n_t + ["r"] * n_r + ["c"] * n_c + ["e"] * n_e
    rng.shuffle(kinds)
    kinds = kinds[: a.max_steps] if a.max_steps else kinds
    steps = len(kinds)
    print(f"[{a.name}] typed rows {len(typed)} eval rows {len(evset)}; steps typed {n_t} roles {n_r} conflict {n_c}"
          f" evidence {n_e}", flush=True)
    if a.dry_run:
        dry_run(a, typed, roles, conf, ev_packs, col_t, col_r, col_e, (n_t, n_r, n_c, n_e), by_src, out_dir)
        return
    model = HoropterModel(a.base, chans=CHANS).to(DEV)
    ans = Answer(model.bert.config.hidden_size, a.arm).to(DEV)
    heads = [p for n, p in model.named_parameters() if not n.startswith("bert.")] + list(ans.parameters())
    body = list(model.bert.parameters())
    print(f"[{a.name}] trainable body params {sum(p.numel() for p in body) / 1e6:.1f}M (full fine-tune)", flush=True)
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": heads, "lr": a.head_lr}],
                            weight_decay=0.01, fused=True)
    warm = max(1, int(a.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    if a.eval_only:   # score a saved checkpoint; same flags as training
        sd = torch.load(a.eval_only, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        ans.load_state_dict(sd["answer"])
        del sd
        kinds = []
        print(f"[{a.name}] eval only: {a.eval_only}", flush=True)

    def typed_stream():
        ep = 0
        while True:
            yield from batches(typed, a.max_blocks, random.Random(a.seed + ep), **bcfg())
            ep += 1

    def stream(rows, bs, seed):
        ep = 0
        while True:
            idx = list(range(len(rows)))
            random.Random(seed + ep).shuffle(idx)
            for i in range(0, len(idx), bs):
                yield [rows[j] for j in idx[i:i + bs]]
            ep += 1

    st, sr, sc = typed_stream(), stream(roles, a.roles_bs, a.seed + 1000), stream(conf, a.conflict_bs, a.seed + 2000)
    se = stream(ev_packs, a.evidence_bs, a.seed + 3000) if ev_packs else None
    curve, t0, acc = [], time.time(), defaultdict(list)
    start = 0
    if a.resume:   # weights, optimiser and schedule from the checkpoint; the data streams are replayed to its step
        ck0 = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck0["model"])
        ans.load_state_dict(ck0["answer"])
        if ck0["opt"] is not None:
            opt.load_state_dict(ck0["opt"])
        else:
            print(f"[{a.name}] resume: checkpoint has no optimiser state, Adam moments restart", flush=True)
        sched.load_state_dict(ck0["sched"])
        start, curve, t0 = ck0["step"], ck0["curve"], time.time() - ck0["sec"]
        for k in kinds[:start]:   # same batches and mismatch draws as the original run, no compute
            if k == "t":
                batch = next(st)
                mismatched_rows(typed, rng, round(len(batch) * a.mismatch), by_src,
                                *((max(state_len(r) for r, _ in batch), batch) if MAX_TOKENS else ()))
            else:
                next(se if k == "e" else sr if k == "r" else sc)
        if rng.getstate() != ck0["rng"]:
            raise SystemExit(f"[{a.name}] resume: replayed data rng differs from the checkpoint (different flags?)")
        torch.set_rng_state(ck0["torch_rng"])
        if DEV == "cuda" and ck0["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck0["cuda_rng"])
        print(f"[{a.name}] resumed from {a.resume} at step {start}/{steps}", flush=True)
        del ck0

    def to_cpu(x):   # deep copy to host memory, so training can go on while the copy is written
        if torch.is_tensor(x):
            return x.detach().to("cpu", copy=True)
        if isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(to_cpu(v) for v in x)
        return copy.deepcopy(x)
    saver = [None]

    def write_ckpt(state, d, f):
        try:
            torch.save(state, f.with_suffix(".tmp"))
            f.with_suffix(".tmp").rename(f)
            for old in sorted(d.glob("step*.pt"))[:-a.keep_ckpt]:
                old.unlink()
            print(f"[{a.name}] checkpoint saved {f} ({f.stat().st_size / 2**30:.1f} GiB)", flush=True)
        except Exception as e:   # a failed save must not kill the run
            print(f"[{a.name}] checkpoint FAILED {f}: {e!r}", flush=True)

    def save_ckpt(step):   # resumable: weights (+ optimiser) snapshotted to RAM, written in a background thread
        if saver[0] is not None:
            saver[0].join()
        d = Path(a.ckpt_dir) if a.ckpt_dir else out_dir / "ckpt"
        d.mkdir(parents=True, exist_ok=True)
        t = time.time()
        state = to_cpu({"step": step, "sec": round(time.time() - t0), "model": model.state_dict(),
                        "answer": ans.state_dict(), "opt": opt.state_dict() if a.save_opt else None,
                        "sched": sched.state_dict(), "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if DEV == "cuda" else None, "curve": curve,
                        "args": vars(a)})
        print(f"[{a.name}] checkpoint step {step} snapshotted in {time.time() - t:.0f}s, writing in background", flush=True)
        saver[0] = threading.Thread(target=write_ckpt, args=(state, d, d / f"step{step:06d}.pt"))
        saver[0].start()

    model.train()
    for step, k in enumerate(kinds[start:], start + 1):
        if k == "t":
            batch = next(st)
            mm = mismatched_rows(typed, rng, round(len(batch) * a.mismatch), by_src,
                                 *((max(state_len(r) for r, _ in batch), batch) if MAX_TOKENS else ()))
            pg = [typed_pack(r["state"], its) for r, its in batch] + [typed_pack(s, its, opts=False) for s, its in mm]
            items = [its for _, its in batch] + [its for _, its in mm]
            res, ul = run_typed(model, ans, col_t, *zip(*pg), items, DEV)
            flat = [it for _, its in batch for it in its]
            l_ans = typed_loss(res, flat)
            if a.typed_stance:
                l_ans = l_ans + a.typed_stance * typed_stance_loss(res, flat, bool(a.stance_undet))
            y = torch.tensor([0.0] * sum(len(its) for _, its in batch) + [1.0] * sum(len(its) for _, its in mm),
                             device=DEV)
            l_rel = F.binary_cross_entropy_with_logits(ul.float(), y)
            loss = l_ans + a.w_rel * l_rel
            acc["ans"].append(l_ans.item())
            acc["relv"].append(l_rel.item())
        else:
            b = col_e(next(se)) if k == "e" else col_r(next(sr) if k == "r" else next(sc))
            if k == "c":   # supervise only the refute (CONFLICT) state channel
                b = JE.conflict_only(b)
            b = to(b, DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                out = model(b)
            loss, l_r, l_e = roles_loss(out, b, CHANS, a.ev_weight, a.loc_weight)
            if k == "c":   # contradiction-only pairs: evidence supervision only; their single label must not train stance
                loss = loss - F.cross_entropy(out["rel"], b["labels"])
            loss = a.w_nli * loss
            acc[f"{k}_rel"].append(l_r.item())
            acc[f"{k}_ev"].append(l_e.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(ans.parameters()), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == steps:
            m = {kk: round(float(np.mean(v)), 4) for kk, v in acc.items() if v}
            acc = defaultdict(list)
            curve.append({"step": step, **m, "sec": round(time.time() - t0)})
            print(f"[{a.name}] step {step}/{steps} {m} {time.time() - t0:.0f}s mem {peak_gb():.1f}G", flush=True)
        if a.save_every and step % a.save_every == 0 and step < steps:
            save_ckpt(step)
    train_sec = round(time.time() - t0)
    if saver[0] is not None:
        saver[0].join()
    del opt, sched
    torch.cuda.empty_cache() if DEV == "cuda" else None
    out_dir.mkdir(parents=True, exist_ok=True)                    # artifacts are kept by default
    if not a.eval_only:
        torch.save({"model": {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, "answer": ans.state_dict()},
                   out_dir / "model.pt")

    res = {"config": vars(a), "temperatures": Ts, "train_rows": len(typed), "steps": steps, "train_sec": train_sec,
           "peak_mem_gb": round(peak_gb(), 1), "curve": curve}
    val = {}
    if evset:   # typed decisions on the held-out pool, and relevance on mismatched states
        P = JE.predict(model, ans, col_t, evset, a.max_blocks * 2, DEV, bcfg())
        Pm = JE.predict(model, ans, col_t, evset, a.max_blocks * 2, DEV, bcfg(), mismatch=True)
        pe = [(it, p) for it, p, _, _ in P]
        res["typed"] = JE.summarise(pe)
        res["typed_extras"] = JE.typed_extras(P, Pm, ans)
        grp = lambda it: "heldout_tasks" if it["heldout"] else "typed_decisions" if it["src"] == "typed_decisions" \
            else "x_nli" if it["src"].startswith("x_") else "in_mixture"   # noqa: E731
        u, um = np.array([x[3] for x in P]), np.array([x[3] for x in Pm])
        val["ans_pool_eval"] = {"all": JE.nll_of(pe), **{g: JE.nll_of([x for x in pe if grp(x[0]) == g])
                                                         for g in ("in_mixture", "heldout_tasks", "typed_decisions", "x_nli")}}
        val["relv"] = round(float(-np.log(1 - u + 1e-9).mean() - np.log(um + 1e-9).mean()) / 2, 4)
        print(f"[{a.name}] typed", json.dumps({k: v for k, v in res["typed"].items() if k != "by_source"}),
              json.dumps(res["typed_extras"]), flush=True)
        with open(out_dir / "eval_preds.jsonl", "w") as f:
            for (it, p, st_, uu), (_, _, _, umm) in zip(P, Pm):
                f.write(json.dumps({"row": it["row"], "qid": it["qid"], "p": [round(float(x), 5) for x in p],
                                    "support": [round(float(x), 5) for x in st_[:, 0]], "p_unrel": round(uu, 5),
                                    "p_unrel_mismatched": round(umm, 5)}) + "\n")
    if a.long_eval:   # held-out long-state split (whole documents, never trained), at this run's state window
        lrows = [json.loads(l) for f in a.long_eval.split(",") for l in open(resolve(root, f))]
        lrows = random.Random(0).sample(lrows, min(a.eval_limit, len(lrows))) if a.eval_limit else lrows
        info = {r["id"]: r for r in lrows}
        PL = JE.predict(model, ans, col_t, JE.eval_set(lrows, a.long_eval_teacher, Ts), a.max_blocks * 2, DEV, bcfg())
        pl = [(it, p) for it, p, _, _ in PL]
        longp = lambda it: info[it["row"]]["state_tokens"] > a.max_state - 2   # noqa: E731
        inw = lambda it: info[it["row"]]["in_window"].get(it["qid"], True)     # noqa: E731
        res["long_eval"] = {"all": JE.summarise(pl)["in_mixture"], "by_source": JE.summarise(pl)["by_source"],
                            **{k: JE.summarise([x for x in pl if f(x[0])])["in_mixture"] for k, f in (
                                ("long_states", longp), ("short_states", lambda it: not longp(it)),
                                ("long_states_in_window", lambda it: longp(it) and inw(it)),
                                ("long_states_outside_window", lambda it: longp(it) and not inw(it)))}}
        val["ans_long_eval"] = JE.nll_of(pl)
        print(f"[{a.name}] long eval", json.dumps({k: v for k, v in res["long_eval"].items() if k != "by_source"}), flush=True)
    rt = resolve(root, a.roles_test) if a.roles_test else None
    if rt and rt.exists():   # evidence roles on e-SNLI test (3 annotators, both sides)
        test = [json.loads(l) for l in open(rt)][: a.eval_limit]
        val.update(JE.roles_val(model, col_r, test, a.ev_weight, a.loc_weight, DEV, a.roles_bs, a.conflict_bs))
        res["roles"] = JE.role_eval(JE.predict_roles(model, col_r, test, DEV), CHANS)
        print(f"[{a.name}] roles", json.dumps(res["roles"]["gold_label"]), json.dumps(res["roles"]["role_detect_auroc"]),
              flush=True)
    res["val"] = val
    print(f"[{a.name}] val", json.dumps(val), flush=True)
    json.dump(res, open(out_dir / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
