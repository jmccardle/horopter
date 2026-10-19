# SPDX-License-Identifier: MIT
"""Typed packs and collators.

A typed question is {"type": noul | choice | score, "instructions": str, "criteria": ...}:
  noul    criteria optional {"true": text, "false": text}; options [true, false]
  choice  criteria {key: description}; options = the keys in order
  score   criteria [level 0 text, level 1 text, ...]; options = "0".."n-1"

A pack = one state + question blocks. Per typed question: one question block (the instructions alone; its relation
head gives relevance) and one isolated block per option ("<instructions> Answer: <option text>").

Collator layout: state = [CLS] state [SEP]; question block i = q [SEP], positions continue after the state.
"""
import bisect
import random

import torch

NOUL_DEFAULT = {"true": "The statement holds.", "false": "The statement does not hold."}


def options(q):
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def option_texts(q):
    names = options(q)
    if q["type"] == "noul":
        cr = q.get("criteria") or NOUL_DEFAULT
        return [f"{n}: {cr.get(n, NOUL_DEFAULT[n])}" for n in names]
    if q["type"] == "score":
        return list(q["criteria"])
    return [n.replace("_", " ") if (d or n) == n else f"{n.replace('_', ' ')}: {d}" for n, d in q["criteria"].items()]


def typed_pack(state, its, opts=True):
    """One pack: per item a question block, then (opts) its option blocks. -> (pack, groups); groups[i] = {"qb": block
    index of item i's question block, "opts": its option block indices}."""
    qs, groups = [], []
    for it in its:
        q = it["q"]
        instr = q.get("instructions") or it["qid"]
        g = {"qb": len(qs)}
        qs.append({"text": instr, "label": 0})
        if opts:
            texts = option_texts(q)
            g["opts"] = list(range(len(qs), len(qs) + len(texts)))
            qs += [{"text": f"{instr} Answer: {o}", "label": 0} for o in texts]
        groups.append(g)
    return {"state": state, "questions": qs}, groups


def n_blocks(its):
    return sum(1 + len(options(it["q"])) for it in its)


def to(b, dev):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


# ------------------------------------------------------------------------------------------------------ batching
def state_len(r):
    return r["_len"][0] if "_len" in r else len(r["state"]) // 3


def pack_len(r, its, max_state):
    """Padded sequence length of a typed pack: state (window-truncated) + its question blocks (prod rows carry exact
    token counts in r['_len']; other rows are estimated from chars)."""
    s, q = r["_len"] if "_len" in r else (len(r["state"]) // 3, 0)
    if not q:
        q = sum(len(it["q"].get("instructions") or "") // 3 + 8 for it in its) * 4
    return min(s, max_state - 2) + 2 + q


def batches(rows, max_blocks, rng, max_tokens=0, max_slots=0, max_state=512):
    """Shuffled batches of (row, items). max_tokens = 0: at most max_blocks blocks per batch; otherwise length-bucketed
    batches under a padded-token budget (see _token_batches)."""
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    if max_tokens:
        yield from _token_batches(rows, idx, max_blocks, max_tokens, max_slots, max_state)
        return
    cur, n = [], 0
    for i in idx:
        nb = n_blocks(rows[i][1])
        if cur and n + nb > max_blocks:
            yield cur
            cur, n = [], 0
        cur.append(rows[i])
        n += nb
    if cur:
        yield cur


def _token_batches(rows, idx, max_blocks, max_tokens, max_slots, max_state, chunk=4096):
    """Shuffled chunks of rows, sorted by pack length inside a chunk, cut into batches that respect max_blocks, the
    padded budget len(batch) * max_len <= max_tokens and (if set) blocks * max_state_len <= max_slots (the [Nq, Smax]
    state gather of the heads); the batch order is shuffled again so lengths do not trend within an epoch."""
    out = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: pack_len(*rows[i], max_state))
        cur, nb, mx, ms = [], 0, 0, 0
        for i in part:
            r, its = rows[i]
            b, ln = n_blocks(its), pack_len(r, its, max_state)
            sl = min(r["_len"][0] if "_len" in r else len(r["state"]) // 3, max_state - 2) + 2
            m2, s2 = max(mx, ln), max(ms, sl)
            if cur and (nb + b > max_blocks or (len(cur) + 1) * m2 > max_tokens or
                        (max_slots and (nb + b) * s2 > max_slots)):
                out.append(cur)
                cur, nb, m2, s2 = [], 0, ln, sl
            cur.append(rows[i])
            nb, mx, ms = nb + b, m2, s2
        if cur:
            out.append(cur)
    random.Random(len(out)).shuffle(out)
    yield from out


# ---------------------------------------------------------------------------------------------------- collators
class Collator:
    """Packs -> tensors, with the state token -> whitespace-word map that evidence targets need."""

    def __init__(self, tok, max_state=256, max_q=64):
        self.tok, self.ms, self.mq = tok, max_state, max_q

    def _state(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"][: self.ms - 2], enc["offset_mapping"][: self.ms - 2]
        spans, i = [], 0
        for w in text.split():                                   # word indices are whitespace words
            s = text.index(w, i)
            spans.append((s, s + len(w)))
            i = s + len(w)
        t2w = []
        for (s, e) in offs:
            ws = [k for k, (a, z) in enumerate(spans) if s < z and e > a]
            t2w.append(ws[-1] if ws else -1)
        return [self.tok.cls_token_id] + ids + [self.tok.sep_token_id], [-1] + t2w + [-1], len(spans)

    def __call__(self, packs):
        B = len(packs)
        st = [self._state(p["state"]) for p in packs]
        qs = [[self.tok(q["text"], add_special_tokens=False)["input_ids"][: self.mq - 1] + [self.tok.sep_token_id]
               for q in p["questions"]] for p in packs]
        T = max(len(s[0]) + sum(len(q) for q in qq) for s, qq in zip(st, qs))
        Smax = max(len(s[0]) for s in st)
        ids = torch.full((B, T), self.tok.pad_token_id)
        pos = torch.zeros((B, T), dtype=torch.long)
        blk = torch.full((B, T), -1)
        q_pack, q_ranges, labels, ev_t, ev_m, w_info, s_m, own = [], [], [], [], [], [], [], []
        for b, ((sid, t2w, nw), qq, p) in enumerate(zip(st, qs, packs)):
            S = len(sid)
            ids[b, :S], pos[b, :S], blk[b, :S] = torch.tensor(sid), torch.arange(S), 0
            o = S
            for i, (q, qd) in enumerate(zip(qq, p["questions"]), 1):
                ids[b, o:o + len(q)] = torch.tensor(q)
                pos[b, o:o + len(q)] = torch.arange(S, S + len(q))
                blk[b, o:o + len(q)] = i
                q_pack.append(b)
                q_ranges.append((b * T + o, len(q), S))
                labels.append(qd["label"])
                tgt, m = torch.zeros(Smax), torch.zeros(Smax)
                words = qd.get("words")
                if words is not None and not (words and isinstance(words[0], list)):   # train: one annotator
                    hw = set(words)
                    for t, w in enumerate(t2w):
                        if w >= 0:
                            m[t] = 1
                            tgt[t] = float(w in hw)
                elif qd.get("unit"):   # composed state: words outside the source unit are known non-evidence
                    u0, u1 = qd["unit"]
                    for t, w in enumerate(t2w):
                        if w >= 0 and not (u0 <= w < u1):
                            m[t] = 1
                # localisation target: tokens of this question's own unit (composed multi-unit states only)
                o_m = torch.zeros(Smax)
                if qd.get("unit") and p.get("n_units", 1) > 1 and qd["label"] < 3:
                    u0, u1 = qd["unit"]
                    for t, w in enumerate(t2w):
                        if u0 <= w < u1:
                            o_m[t] = 1
                sv = torch.zeros(Smax)
                sv[:S] = 1
                ev_t.append(tgt)
                ev_m.append(m)
                own.append(o_m)
                s_m.append(sv)
                w_info.append((t2w, nw))
                o += len(q)
        Nq, Mq = len(q_ranges), max(r[1] for r in q_ranges)
        q_tok = torch.zeros((Nq, Mq), dtype=torch.long)
        q_tok_m = torch.zeros((Nq, Mq))
        for j, (start, m, S) in enumerate(q_ranges):
            q_tok[j, :m] = torch.arange(start, start + m)
            q_tok_m[j, :m] = 1
        s_tok = torch.zeros((B, Smax))
        return {"ids": ids, "pos": pos, "blk": blk, "q_pack": torch.tensor(q_pack), "q_tok": q_tok, "q_tok_m": q_tok_m,
                "s_tok": s_tok, "labels": torch.tensor(labels), "ev_t": torch.stack(ev_t), "ev_m": torch.stack(ev_m),
                "s_m": torch.stack(s_m), "own": torch.stack(own), "w_info": w_info}


class FastCollator(Collator):
    """No state word map (only evidence targets need it, and it is O(tokens x words) in Python). Inference uses this."""

    def _state(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: self.ms - 2]
        return [self.tok.cls_token_id] + ids + [self.tok.sep_token_id], [-1] * (len(ids) + 2), 0


def word_map(tok, text, n_tok):
    """Token -> whitespace-word index (-1 for none) for the first n_tok tokens of `text` (no special tokens)."""
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"][:n_tok]
    sp, i = [], 0
    for w in text.split():
        s = text.index(w, i)
        sp.append((s, s + len(w)))
        i = s + len(w)
    out = []
    for (s, e) in offs:
        ws = [k for k, (a, z) in enumerate(sp) if s < z and e > a]
        out.append(ws[-1] if ws else -1)
    return out, len(sp)


def single(ws):
    """Train-format highlight list (one annotator) or None."""
    return ws if isinstance(ws, list) and not (ws and isinstance(ws[0], list)) else None


class RoleCollator(Collator):
    """Collator plus per-channel evidence targets (es_t / es_m state channels, eq_t / eq_m question channels).
    cross_neg: e-SNLI pairs of the other labels are negatives for every channel."""

    def __init__(self, tok, chans, cross_neg=False, **kw):
        super().__init__(tok, **kw)
        self.chans, self.cross_neg = chans, cross_neg

    def __call__(self, packs):
        b = super().__call__(packs)
        Nq, Smax = b["ev_t"].shape
        Mq = b["q_tok"].shape[1]
        s_ch = [c for c in self.chans if c[1] == "s"]
        q_ch = [c for c in self.chans if c[1] == "q"]
        st, sm = torch.zeros(Nq, Smax, len(s_ch)), torch.zeros(Nq, Smax, len(s_ch))
        qt, qmk = torch.zeros(Nq, Mq, max(len(q_ch), 1)), torch.zeros(Nq, Mq, max(len(q_ch), 1))
        q_w = []
        j = 0
        for p in packs:
            for qd in p["questions"]:
                t2w, _ = b["w_info"][j]
                n_q = int(b["q_tok_m"][j].sum()) - 1                     # question tokens without the final [SEP]
                qw, _ = word_map(self.tok, qd["text"], n_q)
                q_w.append(qw)
                lab, esnli = qd["label"], qd.get("src", p.get("source")) == "esnli"
                for c, (name, _, kind, labels) in enumerate(s_ch):
                    tgt, m = self._target(qd, p, name, kind, labels, lab, esnli, "s")
                    if m is None:
                        continue
                    for t, w in enumerate(t2w):
                        if w >= 0 and (m is True or w not in m):
                            sm[j, t, c] = 1
                            st[j, t, c] = float(w in tgt)
                for c, (name, _, kind, labels) in enumerate(q_ch):
                    tgt, m = self._target(qd, p, name, kind, labels, lab, esnli, "q")
                    if m is None:
                        continue
                    for t, w in enumerate(qw):
                        if w >= 0:
                            qmk[j, t, c] = 1
                            qt[j, t, c] = float(w in tgt)
                j += 1
        b.update(es_t=st, es_m=sm, eq_t=qt, eq_m=qmk, q_w=q_w)
        return b

    def _target(self, qd, p, name, kind, labels, lab, esnli, side):
        """-> (target word set, mask) with mask True = every word, a set = words excluded, None = unsupervised."""
        ws = single(qd.get("words") if side == "s" else qd.get("hwords"))
        if lab == 3:
            # unrelated: no state token is evidence; question side carries no role
            return (set(), True) if side == "s" and qd.get("words") is not None else (None, None)
        if lab in labels and ws:
            return set(ws), True
        if self.cross_neg and esnli and lab in (0, 1, 2) and lab not in labels and (side == "q" or lab < 2 or name != "UNSUPPORTED"):
            return set(), True
        if side == "s" and qd.get("unit") and p.get("n_units", 1) > 1:
            u0, u1 = qd["unit"]                                        # composed state: other units are not evidence
            return set(), set(range(u0, u1))
        return None, None


class LongRoleCollator(RoleCollator):
    """RoleCollator for long states: the same token -> whitespace-word map in O(T log W) by bisect (the base map is
    O(T x W) Python, seconds per 2k-token state). Identical output."""

    def _state(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"][: self.ms - 2], enc["offset_mapping"][: self.ms - 2]
        spans, i = [], 0
        for w in text.split():
            s = text.index(w, i)
            spans.append((s, s + len(w)))
            i = s + len(w)
        starts = [a for a, _ in spans]
        t2w = []
        for (s, e) in offs:   # the last word overlapping (s, e): the last word starting before e, if it ends after s
            k = bisect.bisect_left(starts, e) - 1
            t2w.append(k if k >= 0 and spans[k][1] > s else -1)
        return [self.tok.cls_token_id] + ids + [self.tok.sep_token_id], [-1] + t2w + [-1], len(spans)
