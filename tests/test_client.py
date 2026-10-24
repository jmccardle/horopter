# SPDX-License-Identifier: MIT
"""The bare install: `import horopter` and horopter.connect() work without torch, against any HTTP endpoint."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import horopter


class Echo(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if body.get("model") == "missing":
            code, out = 404, {"detail": "this server does not serve 'missing'"}
        else:
            code, out = 200, {"model": body.get("model") or "horopter", "path": self.path, "body": body,
                              "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": {}}
        self._send(code, out)

    def do_GET(self):
        self._send(200, {"models": [{"name": "horopter"}]} if self.path == "/v1/models" else {"ok": True})

    def _send(self, code, out):
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def url():
    srv = HTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_import_does_not_load_torch():
    code = "import sys, horopter; horopter.connect('http://x'); assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_decide_body_and_routes(url):
    h = horopter.connect(url, model="horopter-typed")
    r = h.decide("s", questions={"q": {"type": "noul"}}, fields={"f": {"type": "text"}}, evidence="chosen",
                 reference_date="2026-10-22")
    assert r["path"] == "/v1/horopter"
    assert r["body"] == {"state": "s", "questions": {"q": {"type": "noul"}}, "fields": {"f": {"type": "text"}},
                         "model": "horopter-typed", "horopter": {"evidence": {"mode": "chosen"}},
                         "reference_date": "2026-10-22"}
    assert h.decide("s", fields={"f": {"type": "text"}}, route="clairic")["path"] == "/v1/clairic"
    assert h.decide("s", questions={"q": {"type": "noul"}}, route="systemone")["path"] == "/v1/systemone"
    assert h.models() == [{"name": "horopter"}] and h.health() == {"ok": True}
    r = h.tokenize({"a": 1})
    assert r["path"] == "/v1/tokenize" and r["body"] == {"state": {"a": 1}, "model": "horopter-typed"}


def test_errors(url):
    with pytest.raises(horopter.HoropterError) as e:
        horopter.connect(url).decide("s", questions={"q": {"type": "noul"}}, model="missing")
    assert e.value.status == 404 and "missing" in str(e.value.detail)


def test_backend_hint():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"horopter\[torch\]"):
            horopter.load("jmccardle/horopter-0.4B")
