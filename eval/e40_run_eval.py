# SPDX-License-Identifier: MIT
"""E40: ask every built set (e40_build_sets.py) against a running server, one model at a time, sequential requests.

    HOROPTER_URL=http://127.0.0.1:8420/v1/systemone python eval/e40_run_eval.py [--data <sets dir>] [--sets ...] \
        [--models horopter,horopter-base,horopter-check] [--limit N] [--raw results/raw]
-> <raw>/<set>__<model>.jsonl (resumable: ids already written are skipped)

One request per row: all questions about a row's state go in the same request (question-invariant answers).
Connection errors (server restart) wait 30 s and retry, up to 20 times. HTTP errors are recorded and skipped.
"""
import argparse
import http.client
import json
import os
import socket
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import ask, URL  # noqa: E402  (honours HOROPTER_URL / MBNLI_URL)

DATA = Path(os.environ.get("HOROPTER_DATA", ".")) / "data/v3/e40"
SETS = ["proofwriter_cwa_d5_dev", "proofwriter_owa_d5_dev", "sharc_dev", "conditionalqa_dev"]
MODELS = ["horopter", "horopter-base", "horopter-check"]
RETRIES, WAIT = 20, 30


def ask_retry(state, questions, model):
    for k in range(RETRIES + 1):
        try:
            return ask(state, questions, model)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException, socket.timeout, TimeoutError) as e:
            if k == RETRIES:
                raise
            print(f"  connection error ({type(e).__name__}: {e}); retry {k + 1}/{RETRIES} in {WAIT}s", flush=True)
            time.sleep(WAIT)


def compact(out):
    ans = {}
    for qid, a in out["answers"].items():
        r = {"noul": a["noul"]} if "noul" in a else {"probabilities": a["probabilities"]}
        m = a.get("mbnli", {})
        r["abstain"] = m.get("abstain", {}).get("p")
        r["relevance"] = m.get("relation", {}).get("relevance")
        ans[qid] = r
    mb = out.get("mbnli", {})
    return ans, {"adapters_used": mb.get("adapters_used"), "truncated": mb.get("truncated"),
                 "input_tokens": out.get("usage", {}).get("input_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="rows per set (0 = all)")
    ap.add_argument("--data", default=str(DATA), help="dir of the built sets (default $HOROPTER_DATA/data/v3/e40)")
    ap.add_argument("--raw", default="results/raw")
    a = ap.parse_args()
    print("server", URL, flush=True)
    for s in a.sets.split(","):
        rows = [json.loads(l) for l in open(Path(a.data) / f"{s}.jsonl")]
        if a.limit:
            rows = rows[:a.limit]
        for m in a.models.split(","):
            out = Path(a.raw) / f"{s}__{m}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
            todo = [r for r in rows if r["id"] not in done]
            print(f"{s} x {m}: {len(done)} done, {len(todo)} to go", flush=True)
            t0, n = time.time(), 0
            with open(out, "a") as f:
                for r in todo:
                    rec = {"id": r["id"], "set": s, "model": m}
                    try:
                        o, ms = ask_retry(r["state"], r["questions"], m)
                        rec["answers"], info = compact(o)
                        rec.update(info, ms=round(ms, 1))
                    except urllib.error.HTTPError as e:
                        rec["error"] = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    n += 1
                    if n % 200 == 0:
                        print(f"  {n}/{len(todo)}  {(time.time() - t0) / n * 1000:.0f} ms/row", flush=True)
            print(f"  finished {n} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
