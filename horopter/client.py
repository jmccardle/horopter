# SPDX-License-Identifier: MIT
"""HTTP client for a Horopter server (or any server speaking the same format). Standard library only; uses httpx
when it is installed (pip install "horopter[client]": connection pooling).

    import horopter
    h = horopter.connect("http://127.0.0.1:8420")         # or horopter.local("jmccardle/horopter-0.4B")
    r = h.decide("Refunds within 30 days. The lamp was bought 12 days ago.",
                 questions={"refund": {"type": "noul", "instructions": "Is a refund allowed?"}},
                 fields={"days": {"type": "duration", "unit": "days"}}, evidence="chosen")
    r["answers"]["refund"]["noul"], r["answers"]["days"]["value"]

The response is the server's JSON body as a dict (docs/API.md). Errors raise horopter.HoropterError(status, detail).
"""
import json
import urllib.error
import urllib.request

ROUTES = {"horopter": "/v1/horopter", "systemone": "/v1/systemone", "clairic": "/v1/clairic"}


class HoropterError(Exception):
    """A 4xx / 5xx answer: status and the server's detail."""
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def build_request(state, questions=None, fields=None, model=None, evidence=None, **extra):
    """The request body decide() sends; extra keys (reference_date, options, horopter, ...) are passed through."""
    body = {"state": state}
    if questions:
        body["questions"] = questions
    if fields:
        body["fields"] = fields
    if model:
        body["model"] = model
    if evidence:
        ext = dict(extra.pop("horopter", None) or {})
        ext["evidence"] = evidence if isinstance(evidence, dict) else {"mode": evidence}
        body["horopter"] = ext
    body.update(extra)
    return body


class Client:
    def __init__(self, url="http://127.0.0.1:8420", model=None, timeout=60.0, headers=None):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.headers = {"content-type": "application/json", **(headers or {})}
        try:
            import httpx
            self._http = httpx.Client(timeout=timeout, headers=self.headers)
        except ImportError:
            self._http = None

    def _call(self, method, path, body=None):
        if self._http is not None:
            r = self._http.request(method, self.url + path, json=body)
            status, text = r.status_code, r.text
        else:
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(self.url + path, data=data, headers=self.headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    status, text = r.status, r.read().decode()
            except urllib.error.HTTPError as e:
                status, text = e.code, e.read().decode()
        try:
            out = json.loads(text)
        except ValueError:
            out = {"detail": text}
        if status >= 400:
            raise HoropterError(status, out.get("detail", out) if isinstance(out, dict) else out)
        return out

    def decide(self, state, questions=None, fields=None, model=None, evidence=None, route="horopter", **extra):
        """Questions (Jev types) and/or fields (Clairic types) about one state -> response dict.
        evidence: None | "chosen" | "all" | {"mode": ..., "threshold": ...}. route: horopter | systemone | clairic."""
        body = build_request(state, questions, fields, model or self.model, evidence, **extra)
        return self._call("POST", ROUTES[route], body)

    def request(self, body, route="horopter"):
        """Send a request body as is (any format the server accepts)."""
        return self._call("POST", ROUTES.get(route, route), body)

    def models(self):
        return self._call("GET", "/v1/models")["models"]

    def health(self):
        return self._call("GET", "/healthz")

    def close(self):
        if self._http is not None:
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Local:
    """The same interface over a model in this process (horopter.local). Needs the torch backend."""

    def __init__(self, service, model=None):
        self.service, self.model = service, model

    def decide(self, state, questions=None, fields=None, model=None, evidence=None, route="horopter", **extra):
        from .service import ServiceError
        body = build_request(state, questions, fields, model or self.model, evidence, **extra)
        try:
            out = self.service.decide(body, clairic=route == "clairic")
        except ServiceError as e:
            raise HoropterError(e.status, e.detail) from None
        return json.loads(json.dumps(out))       # plain JSON types, exactly what the server would send

    def request(self, body, route="horopter"):
        from .service import ServiceError
        try:
            return json.loads(json.dumps(self.service.decide(body, clairic=route == "clairic")))
        except ServiceError as e:
            raise HoropterError(e.status, e.detail) from None

    def models(self):
        return self.service.models()["models"]

    def health(self):
        return self.service.health()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
