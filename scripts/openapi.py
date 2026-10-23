# SPDX-License-Identifier: MIT
"""Write the server's OpenAPI document without loading a model: python scripts/openapi.py > docs/openapi.json"""
import json
from types import SimpleNamespace

from horopter.server import build_app

stub = SimpleNamespace(bank={"base": None}, meta={"base": {}}, config={}, name="stub", run="", device="cpu")
print(json.dumps(build_app(stub).openapi(), indent=1))
