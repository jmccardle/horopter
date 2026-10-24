# SPDX-License-Identifier: MIT
"""Runtime: a trunk with swappable question-path LoRA adapters, typed answering, evidence and abstention features.

    import horopter
    m = horopter.runtime.load("path/to/horopter-0.4B")          # the HF layout written by scripts/export_hf.py
    m.use("base")                                               # or an adapter name from m.bank
    out = m.answer(state, {"q1": {"type": "noul", "instructions": "Is the refund allowed?"}}, evidence=True)

An adapter = LoRA A/B of the trunk (gated to question tokens, so state tokens always see the trunk and one state
encoding is valid for every adapter) + every non-trunk parameter of the model (relation, evidence, region heads) + the
answer head. Swapping copies those tensors in place.

HF layout (a directory):
    config.json                 base model id + base transformers config, dims, max_state, heads, arm, LoRA rank
    model.safetensors           trunk + heads (keys as in training; "answer.*" = the answer head)
    tokenizer files             the base model's tokenizer (optional; else downloaded by base model id)
    adapters/<name>/adapter.safetensors + adapter.json    downloaded only when asked for by name (load(adapters=[...]))
    abstain_head.json           the logistic abstention head (optional)
    quantized/model-<scheme>.safetensors   optional Q8 / Q4 trunks (horopter.quant), loaded with load(..., quant=scheme)
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from . import abstain as A
from . import evidence as E
from .losses import run_typed
from .model import UNREL, Answer, HoropterModel, LoRAGate, LoRALinear, apply_lora, unrel_logit
from .packing import FastCollator, options, to, typed_pack

FORMAT = "horopter-hf-1"


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def target_of(q, gold):
    """Gold {"probabilities": {...}} -> vector in options(q) order (noul: [true, false])."""
    if not gold or "probabilities" not in gold:
        return None
    pr = gold["probabilities"]
    names = options(q)
    if not all(n in pr for n in names):
        return None
    v = np.clip(np.asarray([float(pr[n]) for n in names]), 0, None)
    return v / v.sum() if v.sum() > 0 else None


def items_of(row):
    """One {id, state, questions, gold} row -> typed items (target = soft gold where present)."""
    its = []
    for qid, q in row["questions"].items():
        t = target_of(q, (row.get("gold") or {}).get(qid))
        its.append({"row": row["id"], "qid": qid, "q": q, "type": q["type"], "target": t,
                    "gold": int(np.argmax(t)) if t is not None else None})
    return its


class Student:
    """Trunk P + answer head, with LoRA (B = 0: exactly P until an adapter is loaded) on every trunk Linear."""

    def __init__(self, model, ans, tok, max_state=512, max_q=128, r=16, device=None, name="trunk", run="", config=None):
        device = device or default_device()
        self.device, self.name, self.run, self.r, self.config = device, name, run, r, config or {}
        self.base = (config or {}).get("base_model")
        self.tok = tok
        model.to(device)
        ans.to(device)
        model.float()
        model.lora_gate = LoRAGate()
        self.lora_params = apply_lora(model.bert, r, gate=model.lora_gate)
        self.model, self.ans = model, ans
        self.col = FastCollator(tok, max_state=max_state, max_q=max_q)
        self.base_state = self.adapter_state()   # P itself, servable as the "base" adapter
        self.loaded = "base"
        self.bank = {"base": self.base_state}
        self.meta = {"base": {"name": "base", "kind": "trunk P, no adapter", "p": run}}
        self.source = None       # where the model came from: a local Path or a Ref (hub repo), for adapters by name
        self.sources = {}        # adapter name -> the Ref or Path it was loaded from
        self.abstain_head = None

    @property
    def autocast(self):
        return torch.device(self.device).type == "cuda"

    # ------------------------------------------------------------------ adapters
    def adapter_state(self):
        sd = {}
        for n, mod in self.model.bert.named_modules():
            if isinstance(mod, LoRALinear):
                sd[f"bert.{n}.A"] = mod.A.detach().clone()
                sd[f"bert.{n}.B"] = mod.B.detach().clone()
        for n, p in self.model.named_parameters():
            if not n.startswith("bert."):
                sd[n] = p.detach().clone()
        for n, p in self.ans.named_parameters():
            sd[f"answer.{n}"] = p.detach().clone()
        return sd

    def trainable(self):
        heads = [p for n, p in self.model.named_parameters() if not n.startswith("bert.")] + list(self.ans.parameters())
        return self.lora_params, heads

    @torch.no_grad()
    def load_adapter(self, sd, name="?"):
        mods = dict(self.model.named_modules())
        params = dict(self.model.named_parameters())
        aparams = dict(self.ans.named_parameters())
        for k, v in sd.items():
            if k.startswith("answer."):
                aparams[k[7:]].copy_(v)
            elif k.endswith(".A") or k.endswith(".B"):
                mod = mods[k[:-2]]
                getattr(mod, k[-1]).copy_(v)
            else:
                params[k].copy_(v)
        self.loaded = name

    def add_adapter(self, name, path):
        """Load adapters/<name>/adapter.safetensors (+ adapter.json) into the bank (not activated)."""
        from safetensors.torch import load_file
        path = Path(path)
        self.bank[name] = {k: v.to(self.device) for k, v in load_file(str(path / "adapter.safetensors")).items()}
        meta = path / "adapter.json"
        self.meta[name] = json.load(open(meta)) if meta.exists() else {"name": name}

    def add_adapter_from(self, source, name=None, check_trunk=True):
        """Load an adapter into the bank (see adapter_path for the references: a bundled name, a directory, or a hub
        repo) under name (default: adapter.json "name", else the directory name). check_trunk: refuse an adapter
        trained on another trunk (adapter.json "p" != this model's run); an adapter only fits the exact trunk it was
        trained on. -> the name."""
        path, origin = adapter_path(source, self.source, with_origin=True)
        meta = json.load(open(path / "adapter.json")) if (path / "adapter.json").exists() else {}
        name = name or meta.get("name") or path.name
        if check_trunk and self.run and meta.get("p") and meta["p"] != self.run:
            raise ValueError(f"adapter {name!r} ({source}) was trained on trunk {meta['p']!r}; this model is "
                             f"{self.run!r}. An adapter only fits the trunk it was trained on")
        self.add_adapter(name, path)
        self.sources[name] = origin
        return name

    def ensure(self, ref, check_trunk=True):
        """An adapter name or reference -> its bank name, loading it first if it isn't loaded."""
        if ref in self.bank:
            return ref
        return self.add_adapter_from(ref, check_trunk=check_trunk)

    def use(self, name):
        if self.loaded != name:
            self.load_adapter(self.bank[name], name)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def answer(self, state, questions, evidence=False, max_blocks=96):
        """questions: {qid: typed question}. -> {qid: dict(p, rel [k,3], unrel, f [21], ev [(offsets, [Smax, Cs])])}.
        p is in options(q) order. One pack per <= max_blocks blocks; every question sees the same state."""
        self.model.eval()
        its = [{"row": "req", "qid": qid, "q": q, "type": q["type"], "target": None, "gold": None}
               for qid, q in questions.items()]
        out = {}
        chunks, cur, n = [], [], 0
        for it in its:
            nb = 1 + len(options(it["q"]))
            if cur and n + nb > max_blocks:
                chunks.append(cur)
                cur, n = [], 0
            cur.append(it)
            n += nb
        if cur:
            chunks.append(cur)
        offs = None
        for ch in chunks:
            pack, groups = typed_pack(state, ch)
            b = to(self.col([pack]), self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                o = self.model(b)
            for g, it in zip(groups, ch):
                ul = unrel_logit(o["rel"][g["qb"]: g["qb"] + 1])[0]
                sl = slice(g["opts"][0], g["opts"][-1] + 1)
                rel = self.ans.typed_rel(o["q"][sl], o["rel"][sl])
                z = self.ans.levels(o["q"][sl], rel, it["type"])
                d = {"p": z.float().softmax(-1).cpu().numpy(),
                     "rel": rel[:, :UNREL].float().softmax(-1).cpu().numpy(),
                     "unrel": float(torch.sigmoid(ul))}
                d["f"] = A.features(d, it["type"], o, b, g)
                if evidence:
                    if offs is None:
                        offs = self.state_offsets(state)
                    es = o["ev_s"][sl].float().sigmoid().cpu().numpy()      # [k, Smax, Cs]
                    d["ev"] = [(offs, es[j]) for j in range(es.shape[0])]
                out[it["qid"]] = d
        return out

    @torch.no_grad()
    def answer_one_each(self, pairs, max_packs=32):
        """[(state, question)] with a different state per question (the router's case) -> list of probability vectors,
        several packs per forward pass."""
        self.model.eval()
        out = []
        for i in range(0, len(pairs), max_packs):
            chunk = pairs[i:i + max_packs]
            its = [[{"row": "r", "qid": "q", "q": q, "type": q["type"], "target": None, "gold": None}] for _, q in chunk]
            pg = [typed_pack(s, it) for (s, _), it in zip(chunk, its)]
            res, _ = run_typed(self.model, self.ans, self.col, *zip(*pg), its, self.device)
            out += [z.float().softmax(-1).cpu().numpy() for z, _ in res]
        return out

    def state_offsets(self, state):
        return E.state_offsets(self.tok, state, self.col.ms)

    spans = staticmethod(E.spans)

    def abstain(self, d):
        """Wire-format abstain block of one answer dict."""
        return self.abstain_head.served(d) if self.abstain_head is not None else A.placeholder(d)


def answer_routed(st, state, qs, route, evidence):
    """route {qid: {"adapter": name, "own_spans": bool}} -> {qid: Student.answer dict}. Spans come from the "evidence"
    adapter when one is loaded (its answers are not used), except for own_spans questions, which keep their own
    adapter's spans (field extraction: E43 found the family adapters' spans hit the value more often)."""
    res = {}
    ev_ad = "evidence" in st.bank and bool(evidence)
    own = {q for q in qs if route[q].get("own_spans")}
    for ad in sorted({route[q]["adapter"] for q in qs}):
        sub = {q: qs[q] for q in qs if route[q]["adapter"] == ad}
        st.use(ad)
        res.update(st.answer(state, sub, evidence=bool(evidence) and (not ev_ad or any(q in own for q in sub))))
    others = {q: qs[q] for q in qs if q not in own}
    if ev_ad and others:
        st.use("evidence")
        for qid, d in st.answer(state, others, evidence=True).items():
            res[qid]["ev"] = d["ev"]
    return res


# ---------------------------------------------------------------------------------------------------------- loaders
def build_trunk(base_config, ckpt_model, ckpt_answer, base=None):
    """(transformers config, model state dict, answer state dict) -> (HoropterModel, Answer), weights loaded.
    base: build the encoder with from_pretrained(base) instead of from the config. The weights are overwritten either
    way; this only replays the research construction (and so its torch RNG stream, which the LoRA A init of a newly
    trained adapter draws from). The adapter trainers use it."""
    model = HoropterModel(base=base) if base else HoropterModel(config=base_config)
    model.load_state_dict(ckpt_model)
    ans = Answer(model.bert.config.hidden_size, "hard")
    ans.load_state_dict(ckpt_answer)
    return model, ans


def load_checkpoint(model_pt, base, max_state=512, max_q=128, r=16, device=None, run=None):
    """A training checkpoint (training/train_joint.py: model.pt {"model", "answer"}) -> Student. The base model's
    config and tokenizer come from `base` (a hub id or a local dir); its weights are not needed."""
    sd = torch.load(model_pt, map_location="cpu", weights_only=False)
    model, ans = build_trunk(AutoConfig.from_pretrained(base), sd["model"], sd["answer"])
    del sd
    return Student(model, ans, AutoTokenizer.from_pretrained(base), max_state=max_state, max_q=max_q, r=r,
                   device=device, name=run or Path(model_pt).parent.name, run=run or Path(model_pt).parent.name,
                   config={"base_model": base, "max_state": max_state, "max_q": max_q})


def read_config(path):
    cfg = json.load(open(Path(path) / "config.json"))
    if cfg.get("format") != FORMAT:
        raise ValueError(f"{path}: config.json format {cfg.get('format')!r}, expected {FORMAT!r}")
    return cfg


def base_config_of(cfg):
    bc = dict(cfg["base_config"])
    return AutoConfig.for_model(bc.pop("model_type"), **bc)


class Ref:
    """A hub reference "user/repo[@revision][:name]". name: an adapter in the repo (adapters/<name>/, else the path
    <name>/); none: the adapter at the repo root."""

    def __init__(self, repo, revision=None, sub=""):
        self.repo, self.revision, self.sub = repo, revision, sub.strip("/")

    @classmethod
    def parse(cls, s):
        repo, _, sub = str(s).partition(":")
        repo, _, rev = repo.partition("@")
        return cls(repo, rev or None, sub)

    def __str__(self):
        return self.repo + (f"@{self.revision}" if self.revision else "") + (f":{self.sub}" if self.sub else "")

    def child(self, name):
        """An adapter in the same repo and revision."""
        return Ref(self.repo, self.revision, name)


def _download(ref, patterns):
    from huggingface_hub import snapshot_download   # optional dependency
    return Path(snapshot_download(ref.repo, revision=ref.revision, allow_patterns=patterns))


def adapter_path(source, model=None, with_origin=False):
    """A local adapter directory (adapter.safetensors + adapter.json) for one reference:
        NAME                      one of the model's own adapters, adapters/NAME (fetched from the model's hub repo
                                  on first use; model = Student.source)
        ./dir, /abs/dir           a directory
        user/repo[@rev]           the adapter at a hub repo's root
        user/repo[@rev]:NAME      adapters/NAME in a hub repo laid out like a model repo (else the path NAME/)
    Only the files of that one adapter are downloaded. with_origin: -> (path, the Path or Ref it came from)."""
    out = lambda p, o: (p, o) if with_origin else p
    if isinstance(source, Ref):
        return _from_hub(source, out)
    path = Path(source)
    if (path / "adapter.safetensors").exists():
        return out(path, path)
    if path.exists():
        raise FileNotFoundError(f"{source}: no adapter.safetensors in this directory")
    s = str(source)
    if "/" not in s.partition(":")[0]:               # a bare name: the model's own adapters/NAME
        if model is None:
            raise FileNotFoundError(f"{source}: not a directory, and no model to take a bundled adapter from")
        if isinstance(model, Path):
            p = model / "adapters" / s
            if not (p / "adapter.safetensors").exists():
                have = sorted(d.parent.name for d in (model / "adapters").glob("*/adapter.safetensors"))
                raise FileNotFoundError(f"{source}: the model has no adapters/{s} (it has: {', '.join(have) or 'none'})")
            return out(p, p)
        return adapter_path(model.child(s), with_origin=with_origin)
    return _from_hub(Ref.parse(s), out)


def _from_hub(ref, out):
    if not ref.sub:
        p = _download(ref, ["adapter.*"])
        if not (p / "adapter.safetensors").exists():
            raise FileNotFoundError(f"{ref}: no adapter.safetensors at the repo root (name one: {ref.repo}:NAME)")
        return out(p, ref)
    root = _download(ref, [f"adapters/{ref.sub}/*", f"{ref.sub}/*"])
    for p in (root / "adapters" / ref.sub, root / ref.sub):
        if (p / "adapter.safetensors").exists():
            return out(p, ref)
    raise FileNotFoundError(f"{ref}: no adapter at adapters/{ref.sub}/ or {ref.sub}/")


def adapter_names(model):
    """The adapters a model offers: names under adapters/ (a local Path or a hub Ref; the hub lists files, no
    download)."""
    if isinstance(model, Path):
        return sorted(d.parent.name for d in (model / "adapters").glob("*/adapter.safetensors"))
    from huggingface_hub import list_repo_files
    return sorted({f.split("/")[1] for f in list_repo_files(model.repo, revision=model.revision)
                   if f.startswith("adapters/") and f.endswith("/adapter.safetensors")})


def load(path_or_hub_dir, device=None, adapters=True, abstain_head=True, max_state=None, pretrained_init=False,
         quant=None):
    """Load the HF layout (a local directory, or a hub repo id "user/repo[@revision]" if huggingface_hub is
    installed) -> Student, with the abstention head (if any) in .abstain_head.
    adapters: True = every adapter under adapters/ into .bank; False / None = none; a list = those references
    (bundled names or any adapter_path reference). From the hub only the listed adapters are downloaded, and none with
    False; later Student.ensure(name) fetches more on first use.
    pretrained_init: construct the encoder from the base model's pretrained weights before loading (see build_trunk;
    training only, needs the base weights).
    quant: "w8" | "w4g32" | "w4g128" loads quantized/model-<quant>.safetensors instead of model.safetensors (a hub
    download then fetches only that weights file)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    from . import quant as Q
    wfile = Q.file_of(quant) if quant else "model.safetensors"
    path = Path(path_or_hub_dir)
    source = path
    if not path.exists():
        from huggingface_hub import snapshot_download   # optional dependency
        source = Ref.parse(path_or_hub_dir)
        other = [Q.file_of(s) for s in Q.SCHEMES if s != quant] + (["model.safetensors"] if quant else [])
        other += [] if adapters is True else ["adapters/*"]
        path = Path(snapshot_download(source.repo, revision=source.revision, ignore_patterns=other))
    cfg = read_config(path)
    sd = load_file(str(path / wfile))
    if quant:
        with safe_open(str(path / wfile), "pt") as f:
            sd = Q.unpack(sd, f.metadata() or {})
    model_sd = {k: v for k, v in sd.items() if not k.startswith("answer.")}
    ans_sd = {k[len("answer."):]: v for k, v in sd.items() if k.startswith("answer.")}
    model, ans = build_trunk(base_config_of(cfg), model_sd, ans_sd, cfg["base_model"] if pretrained_init else None)
    del sd
    tok_src = path if (path / "tokenizer_config.json").exists() else cfg["base_model"]
    st = Student(model, ans, AutoTokenizer.from_pretrained(str(tok_src)), max_state=max_state or cfg["max_state"],
                 max_q=cfg.get("max_q", 128), r=cfg["lora"]["r"], device=device, name=cfg.get("name", path.name),
                 run=cfg.get("run", ""), config=cfg)
    st.source = source if isinstance(source, Ref) else path
    st.quant = quant or ""
    if adapters is True:
        if (path / "adapters").is_dir():
            for d in sorted((path / "adapters").glob("*/adapter.safetensors")):
                if d.parent.name != "smoke":
                    st.add_adapter(d.parent.name, d.parent)
                    st.sources[d.parent.name] = d.parent if isinstance(source, Path) else source.child(d.parent.name)
    elif adapters:
        for ref in adapters:
            st.ensure(ref)
    if abstain_head and cfg.get("abstain_head") and (path / cfg["abstain_head"]).exists():
        st.abstain_head = A.AbstainHead.from_json(path / cfg["abstain_head"])
    return st
