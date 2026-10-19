# SPDX-License-Identifier: MIT
"""Evidence adapter: option-level evidence spans for typed questions (noul / choice), on a frozen exported trunk.

Why: the trunk's evidence channels are trained mostly in the roles format (state + hypothesis sentence). On typed
option blocks ("<question> Answer: <option>") they can mislabel spans. This adapter supplies the served spans; the
served answer still comes from the answering adapter.

Data: NLI items whose premise evidence is known, asked as typed questions over the premise:
  e-SNLI train (--roles, source esnli): human premise highlights (words chosen by >= 2 of 3 annotators, else by any),
          labels E / C (N has no premise highlight)
  referent pairs (--e22, gold-consistent, clause conf >= 0.8): premise evidence of supported clauses (S) and
          contradicted clauses (C); held-out test = sha1(id) % 10 == 0
  e-SNLI conflict pairs (--conflict, --n_conflict), generated policies (--n_policy: the deciding rule sentence
          supports the gold option; training/data/policy_gen.py seed 2, --policy_holdout domain left out)
Each NLI item -> one choice question (3 wordings, entailment / neutral / contradiction) and one noul question.
Targets per option block, state tokens (SUPPORT channel = words that make this option true, CONFLICT channel = words
that make it false; everything else in a supervised channel is a negative; unsupervised = masked):
  label E: entailment SUP=S, noul true SUP=S
  label C: contradiction SUP=C, entailment REF=C, noul true REF=C, noul false SUP=C
  --targets mixed: both channels at once on the 'statement holds' option (support = supported-clause evidence,
          refute = contradicted-clause evidence)
Loss = ev_weight * masked BCE on the two state channels + answer CE + typed stance.

    python training/train_evidence.py --trunk export/horopter-0.4B --e22 path/to/referent_pairs.jsonl \
        [--targets mixed --n_conflict 12000 --ref_weight 2 --steps 4000 --n_policy 6000]
-> <out>/adapter.safetensors + adapter.json + eval.json; <out> = --out or $HOROPTER_OUTPUTS/adapters/<trunk>/<name>
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horopter import runtime  # noqa: E402
from horopter.model import REF, SUP, UNREL  # noqa: E402
from horopter.packing import to, typed_pack  # noqa: E402

NLI_CRIT = {"entailment": "the text makes the statement definitely true",
            "neutral": "the text leaves it open (possibly true, possibly false)",
            "contradiction": "the text makes the statement definitely false"}
CHOICE_W = ["What does the text say about the statement «{h}»?", "Statement: «{h}». Given only the text, which holds?",
            "How does the text relate to «{h}»?"]
NOUL_W = ["Does the text establish that «{h}»?", "According to the text, is it true that {h}?", "Is this supported by the text: {h}"]
LAB = {0: "entailment", 1: "contradiction", 2: "neutral"}   # roles-file label ids


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def word_span(text, ws, w):
    """Char span of whitespace word w (the roles files index text.split())."""
    pos = 0
    for i, x in enumerate(ws):
        pos = text.index(x, pos)
        if i == w:
            return (pos, pos + len(x))
        pos += len(x)


def esnli_items(path, n, rng):
    out = []
    for line in open(path):
        r = json.loads(line)
        ws = r["state"].split()
        for q in r["questions"]:
            if (q.get("src") or r.get("source")) != "esnli" or q["label"] not in LAB:
                continue
            lab = LAB[q["label"]]
            if lab == "neutral":
                continue
            if "pwords" in q:      # test file: three annotators
                ann = [set(a) for a in q["pwords"] if a]
                if not ann:
                    continue
                maj = {w for w in set().union(*ann) if sum(w in a for a in ann) >= 2} or set().union(*ann)
            else:                  # train file: one merged highlight
                maj = set(q.get("words") or [])
                if not maj:
                    continue
            spans = [word_span(r["state"], ws, w) for w in sorted(maj) if w < len(ws)]
            out.append({"state": r["state"], "hyp": q["text"], "label": lab,
                        "S": spans if lab == "entailment" else [], "C": spans if lab == "contradiction" else [],
                        "src": "esnli"})
    rng.shuffle(out)
    return out[:n]


def conflict_items(path, n, rng):
    """e-SNLI contradiction pairs with the premise words that conflict: refute-channel supervision on the 'statement
    holds' option, support on 'contradiction'."""
    out = []
    for line in open(path):
        r = json.loads(line)
        if not r.get("gold"):
            continue
        spans = [word_span(r["premise"], r["premise"].split(), w) for w in r["gold"] if w < len(r["premise"].split())]
        out.append({"state": r["premise"], "hyp": r["hyp"], "label": "contradiction", "S": [], "C": spans,
                    "src": "esnli_conflict"})
    rng.shuffle(out)
    return out[:n]


def e22_items(path, n, rng, min_conf=0.8, mode="bw"):
    """-> (train items, held-out test items). Test = sha1(id) % 10 == 0, always in mixed form (the eval targets).
    bw:    E / C items only; S on entailment items, C on contradiction items (one colour per item)
    mixed: every gold-consistent item (E / N / C) with clause evidence; per clause stance: S = supported-clause
           evidence, C = contradicted-clause evidence, D = undetermined / unaddressed evidence (distractors)"""
    tr, te = [], []
    for line in open(path):
        r = json.loads(line)
        lab = (r.get("gold") or {}).get("nli")
        if r.get("gold_consistent") is False or lab not in ("entailment", "contradiction", "neutral"):
            continue
        Sx, Cx, Dx = [], [], []
        for c in r["clauses"]:
            ev = [(e["start"], e["end"]) for e in c["evidence"]]
            if c["stance"] in ("undetermined", "unaddressed"):
                Dx.extend(ev)           # distractors: any confidence (unsettled clauses are the low-confidence ones)
            elif c.get("conf", 0) >= min_conf:
                (Sx if c["stance"] == "supported" else Cx).extend(ev)
        held = int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 10 == 0
        it = {"state": r["state"], "hyp": r["hypothesis"], "label": lab, "S": Sx, "C": Cx, "D": Dx,
              "src": "e22:" + r["dataset"], "mixed": True}
        if held:
            if Sx or Cx:
                te.append(it)
        elif mode == "mixed":
            if Sx or Cx or Dx:
                tr.append(it)
        elif (lab == "entailment" and Sx) or (lab == "contradiction" and Cx):
            tr.append(dict(it, S=Sx if lab == "entailment" else [], D=[], mixed=False))
    rng.shuffle(tr)
    return tr[:n], te


def policy_items(src, n, rng, holdout=None, only=None, eval_states=None):
    """Generated policies (prose register). The 4-option policy question; SUPPORT target on the gold option = the
    deciding rule sentence (gold_span). Items decided by the default (no deciding rule) are skipped.
    src = a policy_gen.py JSONL path, or ("gen", seed): generate fresh items, dropping exact eval_states."""
    if isinstance(src, tuple):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
        import policy_gen as G
        g, gen, i = random.Random(src[1]), [], 0
        while len(gen) < 3 * n:
            gen += G.make(g, 200000 + i)
            i += 1
        seen = eval_states or set()
        gen = [r for r in gen if r["prose_state"] not in seen]
    else:
        gen = [json.loads(l) for l in open(src)]
    out = []
    for r in gen:
        if r.get("gold_span") is None or r["domain"] == holdout or (only and r["domain"] != only):
            continue
        q = r["questions"]["q"]
        gi = list(q["criteria"]).index(r["gold"]["q"])
        out.append({"state": r["prose_state"], "label": "policy", "S": [tuple(r["gold_span"])], "C": [],
                    "src": "e38_policy", "qs": [(q, gi, {gi: ([tuple(r["gold_span"])], None)})]})
    rng.shuffle(out)
    return out[:n]


def to_questions(it, rng):
    """-> list of (question, gold option index in options(q), {option index: (sup spans|None, ref spans|None)})."""
    if "qs" in it:
        return it["qs"]
    h = it["hyp"].rstrip(".")
    lab, Sx, Cx = it["label"], it["S"], it["C"]
    names = list(NLI_CRIT)
    qc = {"type": "choice", "instructions": rng.choice(CHOICE_W).format(h=h), "criteria": dict(NLI_CRIT)}
    qn = {"type": "noul", "instructions": rng.choice(NOUL_W).format(h=h)}
    if it.get("mixed"):
        tc = {names.index("entailment"): (Sx, Cx), names.index("contradiction"): (Cx, None)}
        tn = {0: (Sx, Cx), 1: (Cx, None)}                       # options(noul) = [true, false]
    elif lab == "entailment":
        tc, tn = {names.index("entailment"): (Sx, None)}, {0: (Sx, None)}
    else:
        tc = {names.index("contradiction"): (Cx, None), names.index("entailment"): (None, Cx)}
        tn = {0: (None, Cx), 1: (Cx, None)}
    return [(qc, names.index(lab), tc), (qn, 0 if lab == "entailment" else 1, tn)]


def token_mask(offs, spans):
    """State token positions (pack coordinates) overlapping any char span."""
    m = np.zeros(len(offs), np.float32)
    for t, o in enumerate(offs):
        if o is None or o[1] <= o[0]:
            continue
        if any(o[0] < e and o[1] > s for s, e in spans):
            m[t] = 1.0
    return m


def build(st, items, rng):
    rows = []
    for it in items:
        offs = st.state_offsets(it["state"])
        qs = to_questions(it, rng)
        rows.append({"state": it["state"], "offs": offs, "qs": qs, "src": it["src"], "D": it.get("D") or [],
                     "label": it["label"]})
    return rows


def forward(st, batch):
    """Packed forward over rows -> per question: (z levels, rel, ev_s [k, Smax, Cs], gold, targets, offs, q, row)."""
    packs, groups, flat = [], [], []
    for r in batch:
        its = [{"row": "r", "qid": f"q{i}", "q": q, "type": q["type"], "target": None, "gold": None}
               for i, (q, _, _) in enumerate(r["qs"])]
        p, g = typed_pack(r["state"], its)
        packs.append(p)
        groups.append(g)
        flat.append((r, its))
    b = to(st.col(packs), st.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=st.autocast):
        o = st.model(b)
    out, base = [], 0
    for p, grp, (r, its) in zip(packs, groups, flat):
        for g, it, (q, gold, tgt) in zip(grp, its, r["qs"]):
            sl = slice(base + g["opts"][0], base + g["opts"][-1] + 1)
            rel = st.ans.typed_rel(o["q"][sl], o["rel"][sl])
            z = st.ans.levels(o["q"][sl], rel, it["type"])
            out.append((z, rel, o["ev_s"][sl], gold, tgt, r["offs"], q, r))
        base += len(p["questions"])
    return out


def losses(res, ev_weight, ref_weight=1.0):
    ce, ev = [], []
    for z, rel, es, gold, tgt, offs, q, _ in res:
        k = z.shape[0]
        t = torch.zeros(k, device=z.device)
        t[gold] = 1
        ce.append(-(t * Fn.log_softmax(z.float(), -1)).sum())
        stance_t = torch.stack([t, 1 - t, torch.zeros_like(t)], -1)          # typed stance targets
        ce[-1] = ce[-1] + -(stance_t * Fn.log_softmax(rel[:, :UNREL].float(), -1)).sum(-1).mean()
        L = min(len(offs), es.shape[1])
        for j, (sup, ref) in tgt.items():
            for ch, spans in ((SUP, sup), (REF, ref)):
                if spans is None:
                    continue
                y = torch.from_numpy(token_mask(offs[:L], spans)).to(es.device)
                m = torch.tensor([0.0 if o is None or o[1] <= o[0] else 1.0 for o in offs[:L]], device=es.device)
                l = Fn.binary_cross_entropy_with_logits(es[j, :L, ch].float(), y, reduction="none")
                ev.append((l * m).sum() / m.sum().clamp(min=1) * (ref_weight if ch == REF else 1.0))
    lce = torch.stack(ce).mean()
    lev = torch.stack(ev).mean() if ev else torch.zeros((), device=lce.device)
    return lce + ev_weight * lev, lce.item(), lev.item()


def average_precision_score(y, sc):
    o = np.argsort(-sc)
    y = np.asarray(y)[o]
    hits = np.cumsum(y)
    return float((hits / np.arange(1, len(y) + 1))[y > 0].mean())


@torch.no_grad()
def evaluate(st, rows, bs=16):
    """Token AP of each supervised (option block, channel) target + answer accuracy. On mixed items also
    ap_*_on_<label> and ap_support_vs_distractor (supported-clause tokens vs distractor-clause tokens only)."""
    st.model.eval()
    aps, hit = {}, []
    for i in range(0, len(rows), bs):
        for z, rel, es, gold, tgt, offs, q, r in forward(st, rows[i:i + bs]):
            hit.append(int(z.argmax().item() == gold))
            L = min(len(offs), es.shape[1])
            valid = [t for t in range(L) if offs[t] is not None and offs[t][1] > offs[t][0]]
            for j, (sup, ref) in tgt.items():
                for name, ch, spans in (("support", SUP, sup), ("refute", REF, ref)):
                    if not spans:
                        continue
                    y = token_mask(offs[:L], spans)[valid]
                    sc = es[j, valid, ch].float().cpu().numpy()
                    if 0 < y.sum() < len(y):
                        aps.setdefault(f"ap_{name}", []).append(average_precision_score(y, sc))
                        aps.setdefault(f"ap_{name}_on_{r['label']}", []).append(average_precision_score(y, sc))
                        aps.setdefault(f"top1_{name}", []).append(float(y[int(np.argmax(sc))]))
                    if name == "support" and r["D"]:
                        dm = token_mask(offs[:L], r["D"])[valid] * (1 - y)
                        keep = (y + dm) > 0
                        if y[keep].sum() and dm[keep].sum():
                            aps.setdefault("ap_support_vs_distractor", []).append(average_precision_score(y[keep], sc[keep]))
    st.model.train()
    out = {k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(aps.items())}
    out["answer_acc"] = round(float(np.mean(hit)), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trunk", required=True, help="exported model dir (scripts/export_hf.py)")
    ap.add_argument("--name", default="evidence")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data_root", default=os.environ.get("HOROPTER_DATA", "."))
    ap.add_argument("--roles", default="data/v3/roles/train_roles.jsonl")
    ap.add_argument("--roles_test", default="data/v3/roles/esnli_roles_test.jsonl")
    ap.add_argument("--conflict", default="data/v3/e07/train_conflict.jsonl")
    ap.add_argument("--e22", default=None, help="referent pairs JSONL (clause-level evidence); required if --n_e22 > 0")
    ap.add_argument("--policy_eval", default="data/v3/e38/policy_v0.jsonl",
                    help="policy eval set (policy_gen seed 0): excluded from generated items and scored if present")
    ap.add_argument("--targets", default="bw", choices=["bw", "mixed"], help="referent-pair targets: one colour per "
                                                                           "item, or per clause")
    ap.add_argument("--n_esnli", type=int, default=24000)
    ap.add_argument("--n_e22", type=int, default=8000)
    ap.add_argument("--n_conflict", type=int, default=0, help="extra e-SNLI conflict pairs (refute supervision)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--ev_weight", type=float, default=2.0)
    ap.add_argument("--ref_weight", type=float, default=1.0, help="extra weight on the refute (CONFLICT) channel loss")
    ap.add_argument("--n_policy", type=int, default=0, help="generated-policy deciding-rule spans (policy_gen seed 2)")
    ap.add_argument("--policy_holdout", default="robot_sop")
    ap.add_argument("--max_state", type=int, default=512)
    ap.add_argument("--ref_adapters", default=None, help="dir of served adapters (<name>/adapter.safetensors) scored "
                                                         "before training for reference (typed, evidence)")
    ap.add_argument("--test_n", type=int, default=1500, help="e-SNLI test items scored before and after")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = random.Random(a.seed)
    root = Path(a.data_root)
    rp = lambda p: Path(p) if Path(p).is_absolute() else root / p   # noqa: E731
    trunk = Path(a.trunk)
    out_dir = Path(a.out) if a.out else Path(os.environ.get("HOROPTER_OUTPUTS", "outputs")) / "adapters" / trunk.name / a.name
    st = runtime.load(trunk, adapters=False, abstain_head=False, max_state=a.max_state, pretrained_init=True)
    if a.n_e22 and not a.e22:
        raise SystemExit("--e22 is required with --n_e22 > 0")
    e22_tr, e22_te = e22_items(a.e22, a.n_e22, rng, mode=a.targets) if a.n_e22 else ([], [])
    tr_items = esnli_items(rp(a.roles), a.n_esnli, rng) + e22_tr
    if a.n_conflict:
        tr_items += conflict_items(rp(a.conflict), a.n_conflict, rng)
    pol = rp(a.policy_eval) if a.policy_eval else None
    pol = pol if pol and pol.exists() else None
    if a.n_policy:
        seen = {json.loads(l)["prose_state"] for l in open(pol)} if pol else set()
        tr_items += policy_items(("gen", 2), a.n_policy, rng, holdout=a.policy_holdout, eval_states=seen)
    rng.shuffle(tr_items)
    te_esnli = esnli_items(rp(a.roles_test), a.test_n, random.Random(1))
    train = build(st, tr_items, rng)
    test = build(st, te_esnli, random.Random(2))
    test_mixed = build(st, e22_te, random.Random(3))
    tests = {"esnli": test, "e22_mixed": test_mixed}
    if pol:   # the policy eval set (seed 0): all domains, and the held-out domain alone
        tests["policy_v0"] = build(st, policy_items(str(pol), 10**9, random.Random(4)), random.Random(4))
        tests[f"policy_v0_{a.policy_holdout}"] = build(st, policy_items(str(pol), 10**9, random.Random(4),
                                                                        only=a.policy_holdout), random.Random(4))
    tests = {k: v for k, v in tests.items() if v}
    ev_all = lambda: {k: evaluate(st, v) for k, v in tests.items()}  # noqa: E731
    srcs = {}
    for it in tr_items:
        srcs[it["src"]] = srcs.get(it["src"], 0) + 1
    print(f"[{a.name}] targets {a.targets}; train items {len(train)} {srcs}; test e-SNLI {len(test)}, E22 mixed "
          f"{len(test_mixed)}", flush=True)

    res = {"trunk": trunk.name, "args": vars(a), "train_sources": srcs}
    res["before_base"] = ev_all()   # trunk alone
    print(f"[{a.name}] trunk (no adapter): {res['before_base']}", flush=True)
    ref_dir = Path(a.ref_adapters) if a.ref_adapters else trunk / "adapters"
    for ref in ("typed", "evidence"):   # the served family adapter and the served evidence adapter (if any)
        if (ref_dir / ref / "adapter.safetensors").exists() and ref != a.name:
            st.add_adapter(ref, ref_dir / ref)
            st.use(ref)
            res[f"before_{ref}_adapter"] = ev_all()
            print(f"[{a.name}] served {ref} adapter: {res[f'before_{ref}_adapter']}", flush=True)
            st.load_adapter(st.base_state, "base")

    lora, heads = st.trainable()
    groups = [{"params": heads, "lr": a.head_lr}, {"params": lora, "lr": a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warm = max(1, int(0.06 * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (a.steps - s) / (a.steps - warm))))
    order, curve, run, t0 = [], [], [], time.time()
    st.model.train()
    for step in range(1, a.steps + 1):
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        batch = [train[order.pop()] for _ in range(min(a.bs, len(order)))]
        loss, lce, lev = losses(forward(st, batch), a.ev_weight, a.ref_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run.append((lce, lev))
        if step % 100 == 0 or step == a.steps:
            m = np.mean(run, 0)
            curve.append({"step": step, "ce": round(float(m[0]), 4), "ev": round(float(m[1]), 4),
                          "sec": round(time.time() - t0)})
            run = []
            print(f"[{a.name}] step {step}/{a.steps} ce {m[0]:.4f} ev {m[1]:.4f} {curve[-1]['sec']}s mem {peak_gb():.1f}G",
                  flush=True)
    del opt
    res["after"] = ev_all()
    res["train_sec"] = round(time.time() - t0)
    res["curve"] = curve
    print(f"[{a.name}] evidence adapter: {res['after']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.float().cpu().contiguous() for k, v in st.adapter_state().items()}
    save_file(sd, str(out_dir / "adapter.safetensors"))
    json.dump({"name": a.name, "trunk": trunk.name, "p": st.run, "kind": "loraq+heads (evidence)", "r": st.r,
               "steps": a.steps, "targets": a.targets, "dev_acc": res["after"]["esnli"]["answer_acc"] if "esnli" in
               res["after"] else None, "eval": res["after"], "train_sec": res["train_sec"]},
              open(out_dir / "adapter.json", "w"), indent=1)
    json.dump(res, open(out_dir / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
