# SPDX-License-Identifier: MIT
"""Typed fields (the Clairic request shape) from Horopter's own outputs: "regex proposes, evidence disposes" (E43).

Each field becomes one hidden yes/no question, "Does the text give <the field>?", answered over the cached state with
evidence on. Then:
- p_null      1 - p(yes) of that question (E43: AUROC 0.95-0.99 for absent vs present with the released adapters). With
              the extract adapter, 1 - the peak support score (AUROC 0.995).
- candidates  every match of the type's pattern anywhere in the state (number, date, time, duration), or the support
              spans themselves (text);
- value       the candidate with the most support (max over its tokens of the "yes" option's support channel, plus a
              bonus when the field's keywords label it: the text before it on its line, or its block's "Header:"), parsed
              to the type; null when p_null > null_threshold or nothing matches. `guess` keeps the best candidate when
              the value is null.
- confidence  (1 - p_null) * support of the chosen candidate; p_null when the value is null. Not calibrated.

estimate and forecast need a distribution over values and are not served (`unsupported`).
"""
import datetime as dt
import re

import numpy as np

from .evidence import spans as _spans

SUPPORTED = ("text", "number", "date", "time", "datetime", "duration", "list")
UNSUPPORTED = ("estimate", "forecast")
LIST_ITEMS = ("text", "number", "date")

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?"
         r"|Nov(?:ember)?|Dec(?:ember)?)\.?")
_ORD = r"(?:st|nd|rd|th)?"
DATE = re.compile(
    rf"\b(?P<d1>\d{{1,2}}){_ORD}(?:\s+of)?\s+(?P<m1>{MONTH}),?(?:\s+(?P<y1>\d{{4}}))?\b"
    rf"|\b(?P<m2>{MONTH})\s+(?P<d2>\d{{1,2}}){_ORD}(?:,?\s+(?P<y2>\d{{4}}))?\b"
    r"|\b(?P<iy>\d{4})[-/.](?P<im>\d{1,2})[-/.](?P<id>\d{1,2})\b"
    r"|\b(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<c>\d{4}|\d{2})\b", re.I)
TIME = re.compile(r"\b(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?(?:\s*(?P<ap>[ap])\.?\s?m\b\.?)?"
                  r"|\b(?P<h2>\d{1,2})\s*(?P<ap2>[ap])\.?\s?m\b\.?", re.I)
NUMBER = re.compile(r"(?<![\w.])[-−]?(?:[$€£¥]\s?)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_UNIT_S = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2629800, "year": 31557600}
_UNIT_RE = (r"(?P<u>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)\b")
DURATION = re.compile(rf"\b(?P<n>\d+(?:\.\d+)?)\s*(?:-\s*)?{_UNIT_RE}|\bnet\s+(?P<net>\d+)\b", re.I)
_UNIT_ALIAS = {"sec": "second", "min": "minute", "hr": "hour", "wk": "week", "mo": "month", "yr": "year"}

# Text fields whose instructions (or name) name a known kind of value get that kind's pattern as the candidate set
# (E43's categories); others use the support spans themselves. The keyword found earliest in the instructions wins
# (the head noun comes first: "People to contact about the invoice" is about people); the field name is the fallback.
_NAME_TOK = r"[A-Z\u00C0-\u017F][a-zA-Z'\u2019\u00C0-\u017F-]+"
NAME_RUN = rf"\b{_NAME_TOK}(?:[ \t]+(?:[A-Z]\.|{_NAME_TOK}|van|von|de|da|del|der|di|la|le|bin|al)){{0,3}}\b"
_STOP = {"The", "This", "Dear", "Hi", "Hello", "Mr", "Mrs", "Ms", "Dr", "Please", "Thank", "Thanks", "Regards", "Best",
         "Invoice", "Order", "Customer", "Phone", "Email", "Address", "From", "To", "Date", "Subject", "Ref", "Our", "We",
         "Your", "I", "Kind", "Sincerely", "Contact", "Name", "Guest", "Team", "Hey", "Questions", "Total", "Issued",
         "Payment", "Call", "Ask", "Attn", "Cc", "Signed", "By"}
HINTS = [
    (r"e-?mail", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    (r"phone|telephone|mobile|cell|fax", r"(?:\+|\b)\d[\d ()./-]{5,}\d\b|\+?\(\d[\d ()./-]{5,}\d\b"),
    (r"\burls?\b|website|web site|\blinks?\b", r"https?://[^\s)>\]\"']+|\bwww\.[^\s)>\]\"']+"),
    (r"\bip\b", r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){2,7}\b"),
    (r"invoice|order|tracking|serial|reference|\bref\b|\bpo\b|purchase order|ticket|account|\bids?\b|\bcodes?\b|"
     r"\bsku\b|confirmation|booking", r"\b(?=[A-Za-z0-9/_-]*\d)[A-Za-z0-9][A-Za-z0-9/_-]{2,}\b"),
    (r"\bnames?\b|\bpersons?\b|people|\bcontacts?\b|\bwho\b", NAME_RUN),
]


def hint_of(name, spec):
    for text in (spec.get("instructions") or "", re.sub(r"[_.\-]+", " ", name)):
        hits = [(m.start(), i) for i, (kw, _) in enumerate(HINTS) for m in [re.search(kw, text.lower())] if m]
        if hits:
            return HINTS[min(hits)[1]][1]
    return None


def _clean_name(v):
    """A name-pattern match -> (offset of the first kept word in v, the name with single spaces), or None. Leading
    salutation and label words are dropped; the offset keeps the span on the original text, whatever whitespace (line
    breaks, double spaces) separates the words."""
    toks = list(re.finditer(r"\S+", v))
    while toks and toks[0].group(0).strip(".") in _STOP:
        toks = toks[1:]
    if not toks or re.fullmatch(MONTH, toks[0].group(0)):   # "March 15" is a date, not a person ("Jane" is a name)
        return None
    return toks[0].start(), " ".join(t.group(0) for t in toks)


_KW_STOP = {"the", "a", "an", "of", "to", "for", "on", "in", "at", "by", "and", "or", "is", "are", "was", "be", "this",
            "that", "it", "its", "any", "all", "each", "every", "what", "which", "who", "whom", "how", "when", "where",
            "long", "much", "many", "about", "from", "with", "has", "have", "does", "do", "text", "value", "given",
            "stated", "mentioned", "listed", "written", "field"}
LABEL_W = 0.6     # bonus for a candidate whose label (line prefix or section header) names the field


def keywords(name, spec):
    """Content words of the field's instructions and name, lower case, plural s dropped."""
    words = re.findall(r"[a-z0-9]+", f"{spec.get('instructions') or ''} {re.sub(r'[_.-]+', ' ', name)}".lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _KW_STOP}


def label_of(state, a):
    """What labels the value at char a in a form-like text: the line's text before it, plus the header of its block
    (the nearest earlier line ending with ':' with no blank line in between)."""
    ls = state.rfind("\n", 0, a) + 1
    lab = state[ls:a]
    i = ls - 1
    for _ in range(8):
        if i <= 0:
            break
        ps = state.rfind("\n", 0, i) + 1
        line = state[ps:i].strip()
        if not line:
            break
        if line.endswith(":"):
            lab += " " + line
            break
        i = ps - 1
    return lab


def label_bonus(kw, state, a):
    if not kw:
        return 0.0
    words = {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", label_of(state, a).lower())}
    hit = len(kw & words)
    return LABEL_W * min(1.0, hit / min(2, len(kw)))


def field_question(name, spec):
    """The hidden noul question for one field."""
    ins = (spec.get("instructions") or "").strip().rstrip(".?:").strip()
    what = ins or re.sub(r"[_.\-]+", " ", name).strip()
    if len(what) > 1 and what[0].isupper() and what[1].islower():
        what = what[0].lower() + what[1:]
    if not re.match(r"(the|a|an|any|all|each|every|its|their|his|her|which|what|how|when|where|who)\b", what, re.I):
        what = ("any " if spec.get("type") == "list" else "the ") + what
    return {"type": "noul", "instructions": f"Does the text give {what}?"}


def check(name, spec):
    """-> error string or None."""
    t = spec.get("type")
    if t in UNSUPPORTED:
        return f"field {name!r}: type {t} is not served by this model (it needs a distribution over values)"
    if t not in SUPPORTED:
        return f"field {name!r}: type must be one of {', '.join(SUPPORTED + UNSUPPORTED)}, got {t!r}"
    if t == "list" and spec.get("items") not in LIST_ITEMS:
        return f"field {name!r}: list fields need items = text, number or date"
    nt = spec.get("null_threshold", 0.5)
    if not isinstance(nt, (int, float)) or not 0 < nt <= 1:
        return f"field {name!r}: null_threshold must be in (0, 1]"
    return None


def _score(offs, sup, a, b):
    v = [float(s) for o, s in zip(offs, sup) if o and o[1] > a and o[0] < b]
    return max(v) if v else 0.0


# ------------------------------------------------------------------------------------------------- parsers
def parse_number(s):
    t = s.replace("−", "-").replace(",", "").replace("%", "")
    t = re.sub(r"[$€£¥\s]", "", t)
    try:
        return float(t)
    except ValueError:
        return None


def parse_date(m, ref):
    g = m.groupdict()
    try:
        if g["iy"]:
            return dt.date(int(g["iy"]), int(g["im"]), int(g["id"]))
        if g["a"]:
            a, b, c = int(g["a"]), int(g["b"]), int(g["c"])
            y = c + 2000 if c < 100 else c
            day, mon = (a, b) if a > 12 else (b, a)     # day-first only when the first number can't be a month
            return dt.date(y, mon, day)
        d, mo, y = (g["d1"], g["m1"], g["y1"]) if g["d1"] else (g["d2"], g["m2"], g["y2"])
        mon = _MONTHS.index(mo.lower()[:3]) + 1
        return dt.date(int(y) if y else ref.year, mon, int(d))
    except (ValueError, TypeError):
        return None


def parse_time(m):
    g = m.groupdict()
    h, mi, s = (int(g["h"]), int(g["mi"]), int(g["s"] or 0)) if g["h"] else (int(g["h2"]), 0, 0)
    ap = (g["ap"] or g["ap2"] or "").lower()
    if ap == "p" and h < 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    if h > 23 or mi > 59 or s > 59:
        return None
    return dt.time(h, mi, s)


def parse_duration(m, unit):
    g = m.groupdict()
    if g["net"]:
        secs = int(g["net"]) * _UNIT_S["day"]
    else:
        u = g["u"].lower().rstrip("s")
        u = _UNIT_ALIAS.get(u, u)
        secs = float(g["n"]) * _UNIT_S[u]
    unit = (unit or "seconds").lower().rstrip("s")
    return secs / _UNIT_S[unit] if unit in _UNIT_S else None


# ------------------------------------------------------------------------------------------------- candidates
def candidates(kind, state, offs, sup, spec, ref):
    """-> [(value, start, end, support)] for one item type, strongest first."""
    out = []
    pat = spec.get("_hint") if kind == "text" else None
    if pat:
        for m in re.finditer(pat, state):
            v, a = m.group(0).rstrip(".,;:"), m.start()
            if pat == NAME_RUN:
                c = _clean_name(v)
                if not c or (len(c[1].split()) < 2 and not spec.get("_one_word_names")):
                    continue                             # a person's name has 2+ words unless a first / last name is asked
                if state.rfind("(", 0, a) > state.rfind(")", 0, a) and state.rfind("(", 0, a) > state.rfind("\n", 0, a):
                    continue                             # inside parentheses: a role or note, "(Accounts Payable)"
                a += c[0]
                b, v = m.start() + len(v), c[1]
            else:
                b = a + len(v)
            out.append((v, a, b, _score(offs, sup, a, b)))
    if kind == "text" and not out:
        sv = np.asarray(sup, dtype=float)
        valid = [sv[t] for t, o in enumerate(offs) if o is not None and o[1] > o[0] and t < len(sv)]
        thr = max(0.05, 0.6 * max(valid)) if valid else 0.5
        for sp in _spans(state, offs, sv[:, None], 0, thr=thr, top=8):
            a, b = sp["start"], sp["end"]
            while a < b and state[a] in " \t\n\"'([{":
                a += 1
            while b > a and state[b - 1] in " \t\n.,;:!?\"')]}":
                b -= 1
            if b > a:
                out.append((state[a:b], a, b, sp["weight"]))
    elif kind == "number":
        for m in NUMBER.finditer(state):
            v = parse_number(m.group(0))
            if v is not None:
                out.append((v, m.start(), m.end(), _score(offs, sup, m.start(), m.end())))
    elif kind in ("date", "datetime"):
        for m in DATE.finditer(state):
            d = parse_date(m, ref)
            if d is not None:
                out.append((d, m.start(), m.end(), _score(offs, sup, m.start(), m.end())))
    elif kind == "time":
        for m in TIME.finditer(state):
            t = parse_time(m)
            if t is not None:
                out.append((t, m.start(), m.end(), _score(offs, sup, m.start(), m.end())))
    elif kind == "duration":
        for m in DURATION.finditer(state):
            v = parse_duration(m, spec.get("unit"))
            if v is not None:
                out.append((v, m.start(), m.end(), _score(offs, sup, m.start(), m.end())))
    kw = spec.get("_kw")
    if kw:
        out = [(v, a, b, sc + label_bonus(kw, state, a)) for v, a, b, sc in out]
    out.sort(key=lambda c: -c[3])
    return out


def _format(kind, v, spec):
    if kind in ("number", "duration"):
        if spec.get("integer"):
            return int(round(v))
        dec = spec.get("decimals")
        return round(v, dec) if dec is not None else (int(v) if float(v).is_integer() else v)
    if kind in ("date", "time"):
        return v.isoformat()
    return v


def answer(name, spec, state, offs, sup, p_true, peak_p_null=False, reference_date=None):
    """One field's answer (Clairic answer shape). sup: the "yes" option's support score per state token (aligned with
    offs). peak_p_null: p_null = 1 - peak support (extract adapter) instead of 1 - p(yes)."""
    t = spec["type"]
    spec = dict(spec, _kw=keywords(name, spec))
    if t == "text" or (t == "list" and spec.get("items") == "text"):
        spec["_hint"] = hint_of(name, spec)
        spec["_one_word_names"] = bool(re.search(r"first|given|last|family|surname|nickname",
                                                 f"{spec.get('instructions') or ''} {name}".lower()))
    ref = dt.date.fromisoformat(reference_date) if reference_date else dt.date.today()
    valid = [float(s) for o, s in zip(offs, sup) if o and o[1] > o[0]]
    p_null = 1 - (max(valid) if valid else 0.0) if peak_p_null else 1 - float(p_true)
    p_null = min(1.0, max(0.0, p_null))
    thr = spec.get("null_threshold", 0.5)
    out = {"type": t, "value": None, "p_null": round(p_null, 4)}
    if t == "list":
        kind = spec["items"]
        cs = candidates(kind, state, offs, sup, spec, ref)
        if cs:
            keep_thr = max(0.05, 0.6 * cs[0][3])
            kw = spec.get("_kw")
            top_lab = label_bonus(kw, state, cs[0][1])

            def keep(c):   # strong enough, or labelled like the best item (the rest of a "Contacts:" block)
                return c[3] >= keep_thr or (top_lab > 0 and label_bonus(kw, state, c[1]) >= top_lab)
            seen, items = set(), []
            for v, a, b, s in sorted((c for c in cs if keep(c)), key=lambda c: c[1]):
                fv = _format(kind, v, spec)
                if fv not in seen:
                    seen.add(fv)
                    items.append((fv, a, b, s))
            if items and p_null <= thr:
                out["value"] = [x[0] for x in items]
                out["spans"] = [[x[1], x[2]] for x in items]
                out["confidence"] = round((1 - p_null) * min(1.0, max(x[3] for x in items)), 4)
                return out
            if items:
                out["guess"] = [x[0] for x in items]
        out["confidence"] = round(p_null, 4)
        return out
    if t == "datetime":
        ds = candidates("date", state, offs, sup, spec, ref)
        best = None
        if ds:
            d, a, b, s = ds[0]
            ts = [c for c in candidates("time", state, offs, sup, spec, ref) if -40 <= c[1] - b <= 40 or -40 <= a - c[2] <= 40]
            tm, a2, b2 = (ts[0][0], min(a, ts[0][1]), max(b, ts[0][2])) if ts else (dt.time(0, 0), a, b)
            best = (dt.datetime.combine(d, tm).isoformat(), a2, b2, s)
    else:
        cs = candidates(t, state, offs, sup, spec, ref)
        best = cs[0] if cs else None
        if best is not None:
            best = (_format(t, best[0], spec), best[1], best[2], best[3])
    if best is None:
        out["confidence"] = round(p_null, 4)
        return out
    if p_null > thr:
        out["guess"] = best[0]
        out["confidence"] = round(p_null, 4)
        return out
    out["value"] = best[0]
    out["span"] = [best[1], best[2]]
    out["confidence"] = round((1 - p_null) * min(1.0, best[3]), 4)
    return out
