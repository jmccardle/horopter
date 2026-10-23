# SPDX-License-Identifier: MIT
"""Field extraction (horopter.fields) with a hand-made support signal: runs without model weights."""
import re

import pytest

pytest.importorskip("numpy")
from horopter import fields as F  # noqa: E402

INVOICE = ("Invoice INV-2291 from Northwind Supplies. Issued September 2, 2026. Payment terms: net 30. "
           "Total due: $1,080.00. Questions? Contact Dana Ruiz or Priya Shah at billing@northwind.example.")


def words(state):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", state)]


def support(state, offs, *hot, base=0.02):
    """Support 0.9 on tokens inside any of the given substrings, `base` elsewhere."""
    rng = []
    for h in hot:
        i = state.index(h)
        rng.append((i, i + len(h)))
    return [0.9 if any(o[1] > a and o[0] < b for a, b in rng) else base for o in offs]


def run(spec, *hot, p_true=0.95, name="f", ref="2026-10-22"):
    offs = words(INVOICE)
    return F.answer(name, spec, INVOICE, offs, support(INVOICE, offs, *hot), p_true, reference_date=ref)


def test_question_text():
    assert F.field_question("po_number", {"type": "text"})["instructions"] == "Does the text give the po number?"
    assert F.field_question("x", {"type": "text", "instructions": "The invoice number."})["instructions"] == \
        "Does the text give the invoice number?"
    assert F.field_question("x", {"type": "list", "instructions": "people to contact"})["instructions"] == \
        "Does the text give any people to contact?"


def test_checks():
    assert F.check("a", {"type": "estimate"}) and F.check("a", {"type": "list"}) and F.check("a", {"type": "nope"})
    assert F.check("a", {"type": "list", "items": "number"}) is None


def test_types():
    assert run({"type": "text", "instructions": "The invoice number"}, "INV-2291")["value"] == "INV-2291"
    assert run({"type": "number", "decimals": 2}, "$1,080.00")["value"] == 1080.0
    assert run({"type": "number", "integer": True}, "$1,080.00")["value"] == 1080
    a = run({"type": "date"}, "September 2, 2026")
    assert a["value"] == "2026-09-02" and INVOICE[a["span"][0]:a["span"][1]] == "September 2, 2026"
    assert run({"type": "duration", "unit": "days"}, "net 30")["value"] == 30
    assert run({"type": "duration", "unit": "weeks", "decimals": 2}, "net 30")["value"] == 4.29
    assert run({"type": "text", "instructions": "billing email"}, "Dana", "billing@northwind.example")["value"] == \
        "billing@northwind.example"
    assert run({"type": "list", "items": "text", "instructions": "People to contact"}, "Dana Ruiz or Priya Shah")[
        "value"] == ["Dana Ruiz", "Priya Shah"]


def test_null_and_guess():
    a = run({"type": "text", "instructions": "The PO number"}, "INV-2291", p_true=0.1)
    assert a["value"] is None and a["guess"] == "INV-2291" and a["p_null"] == 0.9
    a = run({"type": "text", "instructions": "The PO number", "null_threshold": 0.95}, "INV-2291", p_true=0.1)
    assert a["value"] == "INV-2291"


def test_parsers():
    ref = __import__("datetime").date(2026, 10, 22)
    got = {m.group(0): F.parse_date(m, ref).isoformat() for m in F.DATE.finditer(
        "2026-01-05, 13/02/2026, 02/03/26, 4th of July, March 9")}
    assert got == {"2026-01-05": "2026-01-05", "13/02/2026": "2026-02-13", "02/03/26": "2026-02-03",
                   "4th of July": "2026-07-04", "March 9": "2026-03-09"}
    assert [F.parse_time(m).isoformat() for m in F.TIME.finditer("at 2:30 pm, 14:05:09 or 12 am")] == \
        ["14:30:00", "14:05:09", "00:00:00"]


def test_names_across_line_breaks():
    state = "Questions? Contact:\nDana  Ruiz\nPriya Shah\ntoday."
    offs = words(state)
    sup = support(state, offs, "Dana  Ruiz\nPriya Shah")
    a = F.answer("contacts", {"type": "list", "items": "text", "instructions": "People to contact"}, state, offs, sup, 0.95)
    assert a["value"] == ["Dana Ruiz", "Priya Shah"]                      # one name per line, never merged
    assert [state[x:y] for x, y in a["spans"]] == ["Dana  Ruiz", "Priya Shah"]
    a = F.answer("who", {"type": "text", "instructions": "The contact name"}, "Contact\nDana Ruiz", [(0, 7), (8, 12), (13, 17)],
                 [0.1, 0.9, 0.9], 0.95)
    assert a["value"] == "Dana Ruiz" and a["span"] == [8, 17]


def test_month_like_names():
    for n in ("Jane Smith", "Marcus Aurelius", "Juno Reyes", "Octavia Butler", "Decker Lane"):
        assert F._clean_name(n) == (0, n)
    assert F._clean_name("March Report") is None and F._clean_name("Sept Totals") is None


def test_form_labels():
    form = ("Contacts:\n- Jane Smith (Accounts Payable) - jane@acme.com\n- Bob Johnson (Procurement) - bob@acme.com\n\n"
            "Billing Email: billing@acme.com\n\nSubtotal: $9,750.50\nTax (8.5%): $828.79\nTotal: $10,579.29\n")
    offs = words(form)
    flat = [0.3] * len(offs)                                   # no evidence preference: the labels decide
    get = lambda n, spec: F.answer(n, spec, form, offs, flat, 0.95)["value"]   # noqa: E731
    assert get("total", {"type": "number", "instructions": "The total amount due"}) == 10579.29
    assert get("email", {"type": "text", "instructions": "The billing email address"}) == "billing@acme.com"
    assert get("contacts", {"type": "list", "items": "text", "instructions": "People to contact"}) == \
        ["Jane Smith", "Bob Johnson"]
