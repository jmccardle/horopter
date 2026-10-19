# SPDX-License-Identifier: MIT
"""E34 shared definitions: pools, teacher roster, source -> family / weight cluster, document groups, hash splits.

Imported by lengths.py and build.py (the E34 CPU data build). Paths resolve under $HOROPTER_DATA (default: the
current directory), the layout the research build used: data/v3/e09/pool_*.jsonl and raw/<teacher>.<pool>.jsonl
(cached teacher answers), data/v3/e33/pool_a25.jsonl, data/v3/e24/source_family.json, data/v3/prod/ (the output).
None of these ship with the repo. Also here: options / gold_index / probs_of (the typed-question helpers of E08).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HOROPTER_DATA", ".")).resolve()
E09 = ROOT / "data/v3/e09"
PROD = ROOT / "data/v3/prod"
POOL_PATH = {"train": E09 / "pool_train.jsonl", "a23": E09 / "pool_a23.jsonl", "a23anli": E09 / "pool_a23anli.jsonl",
             "a25": ROOT / "data/v3/e33/pool_a25.jsonl"}
POOLS = list(POOL_PATH)
TEACHERS = ["jev", "decider", "clefflash"]          # A2.1 roster (user, 2026-10-15)
WINDOW = 2048                                        # serve and train state window (E31)
MAX_Q = 128                                          # FastCollator question-block cap (train_joint default)

# --------------------------------------------------------------------------------------------- families / clusters
# E24 families (horopter/families.py, data/v3/e24/source_family.json) cover the E09 / E14
# sources. The E33 gap-pool sources are new; their E24 family:
A25_FAMILY = {
    "contractnli_doc": "nli", "hover": "nli", "folio": "nli", "control_nli": "nli", "wice_doc": "nli",
    "ragtruth_multi": "nli", "attribution_long": "nli",
    "squad_doc": "knowledge", "qasper_doc": "knowledge", "boolq_doc": "knowledge", "multirc_doc": "knowledge",
    "race_article": "knowledge", "evidence_inference_doc": "knowledge",
    "msmarco_multi": "classification",
    "shp_long": "classification", "ultrafeedback_long": "classification", "hh_rlhf_long": "classification",
    "helpsteer2_long": "classification", "helpsteer3_long": "classification",
}
# Weight clusters for A2.2: the E24 family, with the 55-source classification family split where the teachers
# differ systematically (judge / preference; retrieval / answerability), and the long-document E33 sources as their
# own clusters (their teachers read 2-12k-char states). Each cluster has a parent used when its own fit is too small
# or a teacher has no coverage there yet (e.g. decider / clef-flash on a25 before their runs finish).
JUDGE = {"helpsteer2", "helpsteer3_pref", "hh_rlhf", "shp", "ultrafeedback_pref"}
RETRIEVAL = {"msmarco_rel", "qnli", "wiki_qa"}
CLUSTER_PARENT = {"typed": None, "nli": None, "knowledge": None, "classification": None,
                  "judge": "classification", "retrieval": "classification",
                  "long_nli": "nli", "grounding": "nli", "long_reading": "knowledge", "long_judge": "judge",
                  "long_retrieval": "retrieval"}
A25_CLUSTER = {
    "contractnli_doc": "long_nli", "hover": "long_nli", "folio": "long_nli", "control_nli": "long_nli",
    "wice_doc": "long_nli", "ragtruth_multi": "grounding", "attribution_long": "grounding",
    "squad_doc": "long_reading", "qasper_doc": "long_reading", "boolq_doc": "long_reading",
    "multirc_doc": "long_reading", "race_article": "long_reading", "evidence_inference_doc": "long_reading",
    "msmarco_multi": "long_retrieval",
    "shp_long": "long_judge", "ultrafeedback_long": "long_judge", "hh_rlhf_long": "long_judge",
    "helpsteer2_long": "long_judge", "helpsteer3_long": "long_judge",
}
PREFERENCE_LONG = {"shp_long", "ultrafeedback_long", "hh_rlhf_long", "helpsteer2_long", "helpsteer3_long"}

_SF = None


def family(src):
    global _SF
    if _SF is None:
        _SF = json.load(open(ROOT / "data/v3/e24/source_family.json"))
    if src in A25_FAMILY:
        return A25_FAMILY[src]
    if src.startswith("nli_"):
        return "nli"
    return _SF.get(src, "classification")


def cluster(src):
    if src in A25_CLUSTER:
        return A25_CLUSTER[src]
    if src in JUDGE:
        return "judge"
    if src in RETRIEVAL:
        return "retrieval"
    return family(src)


def chain(cl):
    """Cluster -> [cluster, parent, ..., 'all']."""
    out = []
    while cl:
        out.append(cl)
        cl = CLUSTER_PARENT.get(cl)
    return out + ["all"]


# --------------------------------------------------------------------------------------------- groups / splits

def h01(s, salt=""):
    """Deterministic hash of a string -> [0, 1)."""
    return int(hashlib.md5((salt + s).encode()).hexdigest()[:12], 16) / 16 ** 12


_PREF_CUT = {"shp_long": "\n\nResponse A:", "ultrafeedback_long": "\n\nResponse A:", "hh_rlhf_long": "\n\nResponse A:",
             "helpsteer2_long": "\n\nResponse:", "helpsteer3_long": "\n\nResponse 1:"}
_CHUNKED = {"contractnli_doc", "qasper_doc", "evidence_inference_doc", "boolq_doc", "multirc_doc", "race_article",
            "attribution_long", "control_nli", "folio"}


def group_key(r):
    """Document group of a row: rows of one group share text (chunks of one document, pairs on one prompt) and must
    fall in the same split, and must not be each other's mismatched-state donors."""
    src, rid = r["source"], r["id"]
    key = rid.split("/", 2)[2] if rid.count("/") >= 2 else rid
    if src in _PREF_CUT:
        return src + "|" + hashlib.md5(r["state"].split(_PREF_CUT[src])[0].encode()).hexdigest()[:16]
    if src in _CHUNKED:
        return src + "|" + key.rsplit("/", 1)[0]
    if src == "squad_doc":
        return src + "|" + key.split("/")[0]
    if src in A25_FAMILY:
        return src + "|" + key
    # E09 / E14 pools: one group per state (premise-grouped NLI rows share a state)
    return "s|" + hashlib.md5(r["state"].strip().encode()).hexdigest()[:16]


def fit_bucket(row_id):
    """A2.2 split of the training pools by row: 'fit' (10%) fits temperatures and weights, 'dev' (5%) reports NLL.
    Both stay in training: the fitted parameters are a few numbers per cluster."""
    u = h01(row_id, "e34fit")
    return "fit" if u < 0.10 else "dev" if u < 0.15 else "rest"


def instr_of(q, qid):
    ins = q.get("instructions") or qid
    return ins if isinstance(ins, str) else json.dumps(ins, sort_keys=True)


# --------------------------------------------------------------------------------------------- typed questions (E08)
# Options per question: choice = criteria keys in order; score = levels 0..k-1; noul = [true, false].
# Gold: choice = option name, noul = bool, score = (expected) level, scored as the nearest level.

def options(q):
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def probs_of(q, ans):
    if ans is None:
        return None
    names = options(q)
    if q["type"] == "noul" or "noul" in ans:
        p = float(ans["noul"])
        v = [p, 1 - p]
    else:
        pr = ans["probabilities"]
        if all(n in pr for n in names):
            v = [pr[n] for n in names]
        elif q["type"] == "score" and all(c in pr for c in q["criteria"]):
            v = [pr[c] for c in q["criteria"]]
        else:
            return None
    v = np.clip(np.asarray(v, float), 0, None)
    return v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v))


def gold_index(q, g):
    if g is None:
        return None
    if q["type"] == "noul":
        return 0 if g else 1
    if q["type"] == "score":
        return int(round(float(g)))
    return options(q).index(g) if g in options(q) else None
