# SPDX-License-Identifier: MIT
"""Forward pass over typed packs and the training losses (typed answers, typed stance, roles / evidence)."""
import torch
import torch.nn.functional as F

from .model import UNREL, unrel_logit
from .packing import to


def run_typed(model, ans, col, packs, groups, items, device):
    """-> per item (level logits, relation of its option blocks), and question-block unrelated logits [N_items].
    Packs without option blocks (mismatched packs) contribute question blocks only."""
    b = to(col(list(packs)), device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
        out = model(b)
    res, ul, base = [], [], 0
    for pack, grp, its in zip(packs, groups, items):
        for g, it in zip(grp, its):
            ul.append(unrel_logit(out["rel"][base + g["qb"]: base + g["qb"] + 1])[0])
            if "opts" in g:
                sl = slice(base + g["opts"][0], base + g["opts"][-1] + 1)
                q = out["q"][sl]
                rel = ans.typed_rel(q, out["rel"][sl])
                res.append((ans.levels(q, rel, it["type"]), rel))
        base += len(pack["questions"])
    return res, torch.stack(ul)


def typed_loss(res, its):
    """Soft cross-entropy of each item's level logits against its target distribution, averaged over items."""
    ls = []
    for (z, _), it in zip(res, its):
        t = torch.tensor(it["target"], device=z.device, dtype=torch.float32)
        ls.append(-(t * F.log_softmax(z, -1)).sum())
    return torch.stack(ls).mean()


def typed_stance_loss(res, its, stance_undet=False):
    """Stance targets on typed option blocks: support = the answer target t_k, refute = 1 - t_k (single-answer choice
    and complementary noul: the state supporting another option excludes this one), undetermined = 0. Score
    questions are skipped (their levels are not exclusive hypotheses).
    stance_undet: precomputed [support, refute, undet] rows (it["stance"]) where present, else 1 - t_k is split
    between refute and undetermined by the determinacy max_k t_k."""
    ls = []
    for (_, rel), it in zip(res, its):
        if it["type"] == "score":
            continue
        t = torch.tensor(it["target"], device=rel.device, dtype=torch.float32)
        if stance_undet and it.get("stance") is not None:
            tgt = torch.tensor(it["stance"], device=rel.device, dtype=torch.float32)
        elif stance_undet:
            d = t.max()
            tgt = torch.stack([t, (1 - t) * d, (1 - t) * (1 - d)], -1)
        else:
            tgt = torch.stack([t, 1 - t, torch.zeros_like(t)], -1)
        ls.append(-(tgt * F.log_softmax(rel[:, :UNREL].float(), -1)).sum(-1).mean())
    if ls:
        return torch.stack(ls).mean()
    dev = res[0][1].device if res else "cpu"
    return torch.zeros((), device=dev)


def masked_bce(logit, t, m):
    """Per-channel BCE averaged over supervised tokens; logit/t/m [..., C] -> [C] (0 where unsupervised)."""
    l = F.binary_cross_entropy_with_logits(logit, t, reduction="none") * m
    dims = tuple(range(l.dim() - 1))
    return l.sum(dims) / m.sum(dims).clamp(min=1), m.sum(dims) > 0


def roles_loss(out, b, chans, ev_w, loc_w):
    """Relation CE + ev_w * sum over evidence channels (each channel's BCE averaged over its supervised tokens) +
    loc_w * the unit-contrastive localisation loss. -> (loss, relation CE, evidence term), the last two detached."""
    loss = F.cross_entropy(out["rel"], b["labels"])
    l_rel = loss.detach()
    s_ch = [c for c in chans if c[1] == "s"]
    q_ch = [c for c in chans if c[1] == "q"]
    ls, _ = masked_bce(out["ev_s"], b["es_t"], b["es_m"])
    parts = list(zip(ls, s_ch))
    if q_ch:
        lq, _ = masked_bce(out["ev_q"], b["eq_t"], b["eq_m"])
        parts += list(zip(lq, q_ch))
    l_ev = sum(ev_w * l for l, c in parts)
    loss = loss + l_ev
    sel = b["own"].sum(1) > 0
    if loc_w and sel.any():
        ev, own, sv = out["loc"][sel], b["own"][sel], b["s_m"][sel]
        loss = loss + loc_w * (ev.masked_fill(sv == 0, -1e4).logsumexp(1) - ev.masked_fill(own == 0, -1e4).logsumexp(1)).mean()
    return loss, l_rel, l_ev.detach()
