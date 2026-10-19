# SPDX-License-Identifier: MIT
"""Evidence spans: per-token channel scores of an option block -> whole-word character spans of the state."""


def state_offsets(tok, state, max_state):
    """Char offsets of state token t (position t in the pack: [CLS] ids [SEP], as FastCollator builds it)."""
    enc = tok(state, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"][: max_state - 2]
    return [None] + [tuple(x) for x in off] + [None]


def spans(state, offs, es, chan, thr=None, top=3, min_chars=2):
    """Token scores of one channel (es [Smax, Cs] probabilities) -> up to `top` merged char spans scoring >= thr, by
    max token score, snapped to whole words, returned in text order.
    thr=None: relative, max(0.2, min(0.5, 0.6 * best token score)) (the SUPPORT channel is under-confident at 0.5)."""
    sc = es[: len(offs), chan]
    if thr is None:
        valid = [sc[t] for t, o in enumerate(offs) if o is not None and o[1] > o[0]]
        thr = max(0.2, min(0.5, 0.6 * max(valid))) if valid else 0.5
    found, cur = [], None
    for t, o in enumerate(offs):
        if o is None or o[1] <= o[0]:
            continue
        if sc[t] >= thr:
            if cur and o[0] - cur[1] <= 1:
                cur = [cur[0], o[1], max(cur[2], float(sc[t]))]
            else:
                if cur:
                    found.append(cur)
                cur = [o[0], o[1], float(sc[t])]
        elif cur:
            found.append(cur)
            cur = None
    if cur:
        found.append(cur)
    found = [s for s in found if s[1] - s[0] >= min_chars]
    found.sort(key=lambda s: -s[2])

    def inword(i):   # word characters, plus , . between digits so "$1,000" and "2.5" stay whole
        c = state[i]
        if c.isalnum() or c in "'’-$%":
            return True
        return c in ",." and 0 < i < len(state) - 1 and state[i - 1].isdigit() and state[i + 1].isdigit()

    out, seen = [], set()
    for s, e, w in found:
        while s < e and state[s] == " ":
            s += 1
        while s > 0 and inword(s - 1):   # snap to whole words: a span never starts or ends mid-word
            s -= 1
        while e < len(state) and inword(e):
            e += 1
        if any(s < e2 and s2 < e for s2, e2 in seen):   # overlaps a stronger span after snapping
            continue
        seen.add((s, e))
        out.append({"start": s, "end": e, "text": state[s:e], "weight": round(w, 4)})
        if len(out) == top:
            break
    return sorted(out, key=lambda x: x["start"])
