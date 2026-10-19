# SPDX-License-Identifier: MIT
"""Abstention: 21 scalar features per answered question and the logistic head over them.

Served form: model_uncertainty = the head's P(wrong) from the features; reasons.unrelated = p(unrelated) from the
question block; abstain.p = max of the two. Without a head, a placeholder: model_uncertainty = normalised entropy.
"""
import json
import math

import numpy as np
import torch

from .model import S_CH

FEATURES = ["top1", "top2", "margin", "entropy_norm", "log_n_opt", "is_noul", "is_choice", "is_score",
            "relevance", "stance_sup", "stance_ref", "stance_und", "max_sup", "max_ref", "mean_und",
            "ev_sup_max", "ev_sup_mean", "ev_ref_max", "ev_ref_mean", "ev_sup_share", "ev_ref_share"]


def features(d, qtype, o, b, g):
    """The 21 features of one question, in FEATURES order: answer distribution, relation, and the evidence summary of
    the argmax option over the state tokens. d = {"p", "rel", "unrel"}; o = model output; b = batch; g = the
    question's block group."""
    p, st, u = d["p"], d["rel"], d["unrel"]
    k = int(p.argmax())
    ps = np.sort(p)[::-1]
    p2 = float(ps[1]) if len(ps) > 1 else 0.0
    ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))
    m = b["s_m"][g["opts"][k]].bool()
    ev = torch.sigmoid(o["ev_s"][g["opts"][k]].float())[m]
    es, er = ev[:, S_CH.index("SUPPORT")], ev[:, S_CH.index("CONFLICT")]
    nz = len(es) > 0
    return np.asarray([ps[0], p2, ps[0] - p2, ent, math.log(len(p)), *(float(qtype == t) for t in ("noul", "choice", "score")),
                       1 - u, *st[k], st[:, 0].max(), st[:, 1].max(), st[:, 2].mean(),
                       float(es.max()) if nz else 0.0, float(es.mean()) if nz else 0.0,
                       float(er.max()) if nz else 0.0, float(er.mean()) if nz else 0.0,
                       float((es > 0.5).float().mean()) if nz else 0.0, float((er > 0.5).float().mean()) if nz else 0.0],
                      np.float32)


class AbstainHead:
    """Logistic regression on standardised features: P(wrong) = sigmoid(((f - mu) / sd) . w + b)."""

    def __init__(self, mu, sd, w, b, name="head", meta=None):
        self.mu, self.sd, self.w = (np.asarray(x, np.float32) for x in (mu, sd, w))
        self.b, self.name, self.meta = float(b), name, meta or {}

    @classmethod
    def from_json(cls, path):
        hj = json.load(open(path))
        return cls(hj["mu"], hj["sd"], hj["w"], hj["b"], hj.get("name", "head"),
                   {k: v for k, v in hj.items() if k not in ("mu", "sd", "w", "b")})

    def p_wrong(self, f):
        z = float(((f - self.mu) / self.sd) @ self.w + self.b)
        return 1 / (1 + math.exp(-z))

    def served(self, d):
        """d = one answer dict with "f" and "unrel" -> the wire-format abstain block."""
        mu_ = self.p_wrong(d["f"])
        return {"p": round(max(mu_, d["unrel"]), 4), "model_uncertainty": round(mu_, 4),
                "reasons": {"unrelated": round(d["unrel"], 4), "model_uncertainty": round(mu_, 4)}, "evidence": [],
                "head": self.name}


def placeholder(d):
    """No head loaded: model_uncertainty = normalised entropy of the answer."""
    p = [float(x) for x in d["p"]]
    ent = -sum(x * math.log(x) for x in p if x > 0) / math.log(len(p))
    return {"p": round(max(d["unrel"], ent), 4), "model_uncertainty": round(ent, 4),
            "reasons": {"unrelated": round(d["unrel"], 4)}, "evidence": [],
            "note": "placeholder: no abstention head loaded"}
