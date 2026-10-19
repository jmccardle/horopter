# SPDX-License-Identifier: MIT
"""Problem families (one question-path LoRA adapter each), the router question and what the router reads.

The router is itself an adapter: it answers the hidden choice question ROUTER_Q over router_state(state, question).
The option order of ROUTER_Q is the order the router adapter was trained with; do not reorder.
"""
import json

# family -> (training file stem used by training/build_adapter_data.py, router option description)
FAMILIES = {
    "typed": ("typed_mix", "a business or agent decision about a structured record (noul / choice / score)"),
    "nli": ("nli_mix", "whether a text makes a given statement true, false or undetermined"),
    "relation": ("rel_mix", "how a text relates to a statement, including whether it is unrelated"),
    "classification": ("cls_mix", "labelling a text by topic, sentiment, intent, toxicity, quality or similar"),
    "knowledge": ("kqa_mix", "answering a knowledge, science, reading or commonsense question"),
    "clause": ("clause_gold", "how a text relates to one phrase or clause of a statement"),
}
ROUTER_Q = {"type": "choice", "instructions": "Which kind of task does this question most resemble?",
            "criteria": {k: v for k, (_, v) in FAMILIES.items()}}


def router_state(state, q, max_chars=1500):
    """What the router reads: the asked question (type, instructions, option keys) followed by the start of the state.
    The router answers a question about the question, so the question is the 'state' of the router question."""
    s = state if isinstance(state, str) else json.dumps(state, sort_keys=True, separators=(",", ":"))
    crit = q.get("criteria")
    if isinstance(crit, dict):
        opts = " | ".join(map(str, crit.keys()))
    elif isinstance(crit, list):
        opts = " | ".join(f"level {i}" for i in range(len(crit)))
    else:
        opts = "true | false"
    instr = q.get("instructions")
    instr = instr if isinstance(instr, str) else json.dumps(instr, sort_keys=True)
    return f"Question type: {q['type']}\nQuestion: {instr}\nOptions: {opts}\nText: {s[:max_chars]}"
