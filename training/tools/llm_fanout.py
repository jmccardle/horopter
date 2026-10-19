# SPDX-License-Identifier: MIT
"""llm_fanout.py: fan a JSONL of LLM jobs out across OpenAI-compatible endpoints (llama.cpp llama-server, vLLM, ...).

    python training/tools/llm_fanout.py run --endpoints endpoints.json --in jobs.jsonl --out results.jsonl [--per-board 12]
    python training/tools/llm_fanout.py status --endpoints endpoints.json

Used by the data builders that need a decoder (training/data/cf_pairs.py, check_v2_data.py, compile_real.py): they
write job files, this runs them, they read the results back. Self-hosted endpoints only; nothing here calls a paid API.

Stdlib only (any python >= 3.8). Each input row needs an `id` plus ONE of:
  * `messages`  : OpenAI chat messages  -> POST /v1/chat/completions
  * `prompt`    : raw text              -> POST /completion (llama.cpp native; no chat template)
  * any fields used by --template       -> rendered into one user message, e.g.
                  --template 'Premise: {state}\\nHypothesis: {hypothesis}'
Optional per-row `params` (dict) is merged into the request body last, so a row can set its own
max_tokens, temperature, response_format, logprobs, etc.

Global request options: --system, --max-tokens, --temperature, --json-schema FILE (OpenAI
`response_format: json_schema`, enforced by llama.cpp's grammar sampler), --no-think (sends
chat_template_kwargs.enable_thinking=false, for Qwen3/3.5 hybrids), --top-logprobs K (returns the
top-K logprobs for every generated token: use with --max-tokens 1 for choice readout).

Resumable: rows whose id already has an `ok` record in --out are skipped; failed rows are retried
on the next run (or immediately, up to --retries, on another board). Output is appended and
flushed per row, so a crash loses at most in-flight requests.

Per-board concurrency: every board gets its own worker threads (--per-board N, or `slots` in the
endpoints file); all workers pull from one shared queue, so faster boards take more work. A board
that fails --max-fail times in a row is dropped for the rest of the run and its rows go back on the
queue. Keep N <= the server's `-np` (slot count), otherwise requests just queue in the server.

A "board" is one endpoint (one server). Health: GET /health; llama.cpp's /props and /slots are read when present
(model name, slot count), and a server without /health is accepted if GET /v1/models answers.

Endpoints (required, no default): a JSON file, a list like
    [{"name": "local", "url": "http://localhost:8080", "slots": 16}, ...]
or comma-separated base URLs (e.g. http://localhost:8080,http://localhost:8081).
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------- endpoints
def load_endpoints(path_or_urls, per_board=None):
    if not path_or_urls:
        sys.exit("llm_fanout: --endpoints is required (a JSON file or comma-separated base URLs)")
    if not os.path.exists(path_or_urls) and path_or_urls.startswith("http"):
        eps = [{"name": u.split("//")[1], "url": u} for u in path_or_urls.split(",")]
    else:
        with open(path_or_urls) as f:
            eps = json.load(f)
    for e in eps:
        e["url"] = e["url"].rstrip("/")
        if per_board:
            e["slots"] = per_board
        e.setdefault("slots", 4)
    return eps


def http_json(url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def board_info(ep, timeout=5):
    out = {"name": ep["name"], "url": ep["url"]}
    try:
        try:
            h = http_json(ep["url"] + "/health", timeout=timeout)
            out["health"] = h.get("status", "ok") if isinstance(h, dict) else "ok"
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            models = http_json(ep["url"] + "/v1/models", timeout=timeout)   # no /health: an OpenAI-style server
            out["health"] = "ok"
            out["model"] = ",".join(m.get("id", "?") for m in models.get("data", []))
        try:   # llama.cpp extras (model path, slot count, busy slots)
            props = http_json(ep["url"] + "/props", timeout=timeout)
            out["model"] = os.path.basename(props.get("model_path", "?"))
            out["slots"] = props.get("total_slots")
            out["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
            slots = http_json(ep["url"] + "/slots", timeout=timeout)
            out["busy"] = sum(1 for s in slots if s.get("is_processing"))
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        out["health"] = "DOWN (%s)" % (e.__class__.__name__,)
    return out


# ----------------------------------------------------------------------------- requests
def build_request(row, a):
    """Return (path, body) for one job row."""
    params = dict(row.get("params") or {})
    if "prompt" in row and "messages" not in row and not a.template:
        body = {"prompt": row["prompt"], "n_predict": a.max_tokens, "temperature": a.temperature,
                "cache_prompt": True}
        if a.json_schema_obj is not None:
            body["json_schema"] = a.json_schema_obj
        if a.top_logprobs:
            body["n_probs"] = a.top_logprobs
        body.update(params)
        return "/completion", body
    if "messages" in row:
        messages = list(row["messages"])
    else:
        messages = [{"role": "user", "content": a.template.format(**row).replace("\\n", "\n")}]
    if a.system and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": a.system}] + messages
    body = {"messages": messages, "max_tokens": a.max_tokens, "temperature": a.temperature,
            "cache_prompt": True}
    if a.json_schema_obj is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "out", "schema": a.json_schema_obj}}
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if a.top_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = a.top_logprobs
    body.update(params)
    return "/v1/chat/completions", body


def parse_response(path, resp, want_json):
    rec = {}
    if path == "/completion":
        rec["text"] = resp.get("content", "")
        t = resp.get("timings", {})
        rec["usage"] = {"prompt_tokens": resp.get("tokens_evaluated"), "completion_tokens": t.get("predicted_n")}
        if "completion_probabilities" in resp:
            rec["logprobs"] = resp["completion_probabilities"]
    else:
        ch = resp["choices"][0]
        msg = ch.get("message", {})
        rec["text"] = msg.get("content") or ""
        if msg.get("reasoning_content"):
            rec["reasoning"] = msg["reasoning_content"]
        rec["finish_reason"] = ch.get("finish_reason")
        rec["usage"] = resp.get("usage")
        t = resp.get("timings", {})
        if ch.get("logprobs"):
            rec["logprobs"] = ch["logprobs"].get("content")
    if t:
        rec["timings"] = {k: round(t[k], 2) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                                      "predicted_per_second", "cache_n") if k in t}
    if want_json:
        try:
            rec["json"] = json.loads(rec["text"])
        except Exception:  # noqa: BLE001
            rec["json_error"] = True
    return rec


# ----------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, a, endpoints, rows, out_path):
        self.a, self.endpoints, self.out_path = a, endpoints, out_path
        self.q = queue.Queue()
        for r in rows:
            self.q.put((r, 0))
        self.total = len(rows)
        self.lock = threading.Lock()
        self.done = self.ok = self.failed = 0
        self.tok_out = self.tok_in = 0
        self.lat = []
        self.per_board = {e["name"]: 0 for e in endpoints}
        self.consec_fail = {e["name"]: 0 for e in endpoints}
        self.dead = set()
        self.records = []
        self.t0 = time.monotonic()
        self.out = open(out_path, "a") if out_path else None

    def worker(self, ep):
        a = self.a
        while ep["name"] not in self.dead:
            try:
                row, tries = self.q.get(timeout=0.5)
            except queue.Empty:
                if self.done >= self.total:
                    return
                continue
            path, body = build_request(row, a)
            t = time.monotonic()
            try:
                resp = http_json(ep["url"] + path, body, timeout=a.timeout)
                rec = parse_response(path, resp, a.json_schema_obj is not None)
                rec.update(id=row["id"], board=ep["name"], status="ok", latency_s=round(time.monotonic() - t, 3))
                with self.lock:
                    self.consec_fail[ep["name"]] = 0
            except Exception as e:  # noqa: BLE001
                err = "%s: %s" % (e.__class__.__name__, getattr(e, "reason", "") or str(e)[:300])
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        err += " " + e.read().decode()[:300]
                    except Exception:  # noqa: BLE001
                        pass
                with self.lock:
                    self.consec_fail[ep["name"]] += 1
                    if self.consec_fail[ep["name"]] >= a.max_fail and ep["name"] not in self.dead:
                        self.dead.add(ep["name"])
                        print("llm_fanout: dropping board %s after %d consecutive failures (%s)"
                              % (ep["name"], a.max_fail, err), file=sys.stderr)
                if tries + 1 < a.retries and len(self.dead) < len(self.endpoints):
                    self.q.put((row, tries + 1))
                    continue
                rec = {"id": row["id"], "board": ep["name"], "status": "error", "error": err,
                       "latency_s": round(time.monotonic() - t, 3)}
            for k in a.keep:
                if k in row:
                    rec.setdefault("row_" + k, row[k])
            self.emit(rec)

    def emit(self, rec):
        with self.lock:
            self.done += 1
            self.per_board[rec["board"]] += 1
            if rec["status"] == "ok":
                self.ok += 1
                self.lat.append(rec["latency_s"])
                u = rec.get("usage") or {}
                self.tok_out += u.get("completion_tokens") or 0
                self.tok_in += u.get("prompt_tokens") or 0
            else:
                self.failed += 1
            if self.out:
                self.out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.out.flush()
            if self.a.collect:
                self.records.append(rec)

    def progress(self):
        el = time.monotonic() - self.t0
        return ("%d/%d done (%d ok, %d err) %.1fs  %.2f rows/s  %.1f out-tok/s  %s"
                % (self.done, self.total, self.ok, self.failed, el, self.done / max(el, 1e-9),
                   self.tok_out / max(el, 1e-9), " ".join("%s=%d" % kv for kv in self.per_board.items())))

    def run(self):
        threads = []
        for ep in self.endpoints:
            for _ in range(ep["slots"]):
                th = threading.Thread(target=self.worker, args=(ep,), daemon=True)
                th.start()
                threads.append(th)
        last = time.monotonic()
        while any(th.is_alive() for th in threads):
            time.sleep(0.2)
            if len(self.dead) == len(self.endpoints):
                print("llm_fanout: all boards down, stopping", file=sys.stderr)
                break
            if not self.a.quiet and time.monotonic() - last > self.a.progress_every:
                print("llm_fanout: " + self.progress(), file=sys.stderr)
                last = time.monotonic()
        if self.out:
            self.out.close()
        el = time.monotonic() - self.t0
        lat = sorted(self.lat)
        pct = (lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None)
        return {"rows": self.done, "ok": self.ok, "errors": self.failed, "wall_s": round(el, 2),
                "rows_per_s": round(self.done / max(el, 1e-9), 3),
                "out_tok_per_s": round(self.tok_out / max(el, 1e-9), 1),
                "in_tok_per_s": round(self.tok_in / max(el, 1e-9), 1),
                "lat_p50": pct(0.5), "lat_p95": pct(0.95), "per_board": dict(self.per_board),
                "dead": sorted(self.dead)}


def done_ids(out_path):
    ids = set()
    if out_path and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("status") == "ok":
                    ids.add(r["id"])
    return ids


def add_common(p):
    p.add_argument("--endpoints", default=None, help="endpoints JSON file, or comma-separated base URLs (required)")
    p.add_argument("--per-board", type=int, default=None, help="concurrent requests per board (overrides slots)")
    p.add_argument("--system", default=None)
    p.add_argument("--template", default=None, help="python format string over row fields -> user message")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--json-schema", default=None, help="JSON schema file; output constrained by grammar")
    p.add_argument("--no-think", action="store_true", help="chat_template_kwargs.enable_thinking=false")
    p.add_argument("--top-logprobs", type=int, default=0)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-fail", type=int, default=5, help="consecutive failures before a board is dropped")
    p.add_argument("--keep", default="", help="comma list of input fields copied into the output as row_<field>")
    p.add_argument("--progress-every", type=float, default=15)
    p.add_argument("--quiet", action="store_true")


def finish_args(a):
    a.json_schema_obj = json.load(open(a.json_schema)) if a.json_schema else None
    a.keep = [k for k in a.keep.split(",") if k]
    return a


def run_rows(rows, endpoints, out_path=None, collect=True, **kw):
    """Library entry point: run rows (list of dicts) and return (stats, records)."""
    p = argparse.ArgumentParser()
    add_common(p)
    a = p.parse_args([])
    for k, v in kw.items():
        setattr(a, k.replace("-", "_"), v)
    a.json_schema_obj = kw.get("json_schema_obj")
    a.keep = kw.get("keep", [])
    a.collect = collect
    a.quiet = kw.get("quiet", True)
    r = Runner(a, endpoints, rows, out_path)
    return r.run(), r.records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="run a JSONL of jobs")
    pr.add_argument("--in", dest="inp", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--limit", type=int, default=0)
    add_common(pr)
    ps = sub.add_parser("status", help="health/model/slots of every board")
    ps.add_argument("--endpoints", default=None, help="endpoints JSON file, or comma-separated base URLs (required)")
    a = p.parse_args()

    if a.cmd == "status":
        for ep in load_endpoints(a.endpoints):
            print(json.dumps(board_info(ep)))
        return

    finish_args(a)
    a.collect = False
    endpoints = load_endpoints(a.endpoints, a.per_board)
    live = []
    for ep in endpoints:
        info = board_info(ep)
        print("llm_fanout: %s %s %s slots=%s -> using %d workers" % (ep["name"], info.get("health"), info.get("model"),
                                                            info.get("slots"), ep["slots"]), file=sys.stderr)
        if info.get("health") == "ok":
            live.append(ep)
    if not live:
        sys.exit("llm_fanout: no healthy boards")
    with open(a.inp) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    seen, skip = set(), done_ids(a.out)
    todo = []
    for r in rows:
        if "id" not in r:
            sys.exit("llm_fanout: every row needs an `id`")
        if r["id"] in seen:
            sys.exit("llm_fanout: duplicate id %r" % r["id"])
        seen.add(r["id"])
        if r["id"] not in skip:
            todo.append(r)
    print("llm_fanout: %d rows, %d already done, %d to run on %d boards" % (len(rows), len(rows) - len(todo), len(todo), len(live)),
          file=sys.stderr)
    if not todo:
        return
    stats = Runner(a, live, todo, a.out).run()
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
