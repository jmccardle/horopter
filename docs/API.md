# Horopter HTTP API

One server, one request shape, one response shape. The request is a superset of two published APIs:

- **Jev's System One:** typed *questions* about a state. A question is `noul` (yes/no), `choice` (pick one) or `score` (a level on a scale).
- **Cloudflare's Clairic:** typed *fields* extracted from a state: `text`, `number`, `date`, `time`, `datetime`, `duration` or `list`.

A request can carry questions, fields, or both, over one cached state. A Jev client or a Clairic client gets the body it expects, because the response is a superset of both answers. Every answer can carry the extras those APIs don't have: evidence spans, relation (relevance × stance) and a calibrated abstention probability.

The machine-readable spec is [`openapi.json`](openapi.json). A running server also serves it at `/openapi.json`, with a browsable version at `/docs`.

## Start a server

```bash
pip install "horopter[server]"
python -m horopter.server --model jmccardle/horopter-0.4B --device cpu --port 8420 \
    --router router --adapters evidence,extract,check2                                  # add --fast for the state cache
```

Without flags the server loads the trunk alone and downloads no adapters (docs/ADAPTERS.md). The examples below use the bundled worked example, the router over six family adapters, plus the general helpers.

The first start downloads the model, about 0.8 GB, plus the adapters named (40 MB each). Later starts load in about 12 s on a CPU. Every example below uses `curl` against `http://127.0.0.1:8420`; the outputs shown are real, from Horopter-0.4B on a CPU.

## From Python

`pip install horopter` is enough to call a server; `horopter.local(...)` (with `[torch]`) runs the same requests in your own process and returns the same dicts.

```python
import horopter
h = horopter.connect("http://127.0.0.1:8420")       # or horopter.local("jmccardle/horopter-0.4B", device="cpu")
r = h.decide(state, questions={...}, fields={...}, evidence="chosen", reference_date="2026-10-22")
h.decide(state, fields={...}, route="clairic")       # route: horopter (default) | systemone | clairic
h.models(); h.health()
```

Errors raise `horopter.HoropterError` with `.status` and `.detail`, whether local or remote.

## Routes

| route | for | body |
|---|---|---|
| `POST /v1/systemone` | Jev clients | the shared response |
| `POST /v1/clairic` | Clairic clients | the shared response |
| `POST /ai/run/@cf/cloudflare/clairic` | Workers AI REST clients | `{"result": <shared response>, "success": true, "errors": [], "messages": []}` |
| `POST /client/v4/accounts/{id}/ai/run/@cf/cloudflare/clairic` | the same, with Cloudflare's account path | as above |
| `POST /v1/horopter` | our own name for the shared handler | the shared response |
| `POST /v1/tokenize` | the state as the model reads it: rendered text, token count, limit, each token's character offsets | see [Token counts and offsets](#token-counts-and-offsets) |
| `GET /v1/models` | the model names this server answers to | |
| `GET /healthz` | liveness, loaded adapters, version, `max_state_tokens`, `max_batch` | |

Every request object is checked for unknown keys: a misspelt key (`"evidnce"`, a question's `"question"` instead of `"instructions"`) is a 400 that names it, never silently dropped.

A request can carry any number of questions and fields. The server answers them in passes of `--max_batch` (default 32) over the same cached state, so the cost grows linearly and a large request does not run out of memory.

The routes differ in two ways only:
- The Clairic routes read the first window of a state that is too long and set `truncated: true`. The others refuse with a 413 unless you send `options.allow_truncation: true`.
- The Clairic routes require a non-empty state.

## Step 1: one yes/no question

```bash
curl -s localhost:8420/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Refunds within 30 days. The lamp was bought 12 days ago.",
  "questions": {"refund": {"type": "noul", "instructions": "Is a refund allowed?"}}
}'
```

```json
{"model": "horopter",
 "usage": {"input_tokens": 19, "output_tokens": 0, "state_tokens": 14, "image_tokens": 0, "schema_tokens": 5, "passes": 1},
 "answers": {"refund": {"type": "noul", "noul": 0.9433,
   "horopter": {"relation": {"true": {"support": 0.9207, "refute": 0.0793, "undetermined": 0.0},
                             "false": {"support": 0.0456, "refute": 0.9544, "undetermined": 0.0},
                             "relevance": 0.9996},
                "abstain": {"p": 0.0647, "model_uncertainty": 0.0647,
                            "reasons": {"unrelated": 0.0004, "model_uncertainty": 0.0647}, "evidence": [], "head": "lr_T2"}}}},
 "truncated": false,
 "horopter": {"version": "3.1.0+contract2 trunk=horopter-0.4b-20261020:prod_e400m_b2", "state_id": "sha256:c9b6…",
              "server_ms": 602.8, "queue_ms": 0.0, "adapters_used": ["classification"]}}
```

How to read it:
- **`noul`** is P(yes): 0.94.
- **`horopter.relation`:** the state supports "true" (0.92) and refutes "false" (0.95). `relevance` 0.9996 says the state is about the question.
- **`horopter.abstain.p`:** the probability that this answer is wrong, 0.06. Use it as a threshold: above it, ask a human or a bigger model. `reasons.unrelated` is high when the state doesn't bear on the question.
- **The top-level `horopter` block:** the model version, a hash of the state (`state_id`) and timings.

## Step 2: several questions about one state

The state is encoded once. Each further question costs only its own short pass, and adding a question never changes the others' answers.

```bash
curl -s localhost:8420/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Refunds within 30 days. The lamp was bought 12 days ago.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund allowed?"},
    "dept":   {"type": "choice", "instructions": "Which team handles this?",
               "criteria": {"billing": "billing and refunds", "tech": "technical support"}},
    "size":   {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]}}
}'
```

The Jev fields per type:
- **`choice`:** `choice`, plus `confidence` (rescaled so that chance is 0) and `probabilities` per option. Here it returned `{"choice": "billing", "confidence": 0.814, "probabilities": {"billing": 0.907, "tech": 0.093}}`.
- **`score`:** `score` (the expected level), `confidence`, `legend` and `probabilities` per level. `horopter.score.median` is the median level.

## Step 3: evidence spans

Ask for evidence and each answer lists the words in the state that support or refute each option, as character offsets.

```bash
curl -s localhost:8420/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Refunds within 30 days. The lamp was bought 12 days ago.",
  "questions": {"refund": {"type": "noul", "instructions": "Is a refund allowed?"}},
  "horopter": {"evidence": {"mode": "chosen", "threshold": 0.5}}
}'
```

`evidence` can also be just the mode (`"horopter": {"evidence": "chosen"}`), and a top-level `"evidence": "chosen"` (or the full object) is the same option, shorter. Sending both, differently, is a 400.

- **`mode`:**
  - `chosen`: spans for the winning option only;
  - `all`: spans for every option;
  - `off`: no spans (the default).
- **Each span:** `{start, end, text, weight, side: "state", role: "support" | "refute", option}`.
- Evidence works on text states. JSON states are answered, but without spans. To get spans over JSON, render it yourself (`json.dumps(x, sort_keys=True, separators=(",", ":"))`, the server's own rendering) and send the string.

### Token counts and offsets

Span offsets (`start`, `end`) are character offsets into the **rendered state**: a string is used as is, JSON is rendered with sorted keys and no spaces. `POST /v1/tokenize` shows that text and how the model reads it, before you ask anything:

```bash
curl -s localhost:8420/v1/tokenize -H 'content-type: application/json' -d '{"state": "Refunds within 30 days."}'
# {"state_text": "Refunds within 30 days.", "state_tokens": 6, "limit": 2046, "truncated": false, "read_chars": 23,
#  "offsets": [[0, 3], [3, 7], [7, 14], [14, 17], [17, 22], [22, 23]], "state_id": "sha256:7294ffb1…"}
```

- `state_tokens`: tokens in the whole rendered state, the same count as `usage.state_tokens` (which is capped at `limit`).
- `limit`: tokens the model reads (`max_state_tokens` in `/healthz`). A longer state is a 413 unless truncation is allowed.
- `read_chars`: characters of `state_text` inside the window: all of it, or where a truncated state is cut.
- `offsets`: `[start, end)` per token (a token's span can include its leading space). Evidence spans are built from these tokens and then snapped to whole words.
- `state_id`: the id `decide` reports for this state, so a client can tell which cached state an answer used.
- An optional `model` is checked like `decide`'s (404 if not served); all served models share one tokenizer.

From Python: `h.tokenize(state)` on either client.

## Step 4: extract fields (Clairic)

Clairic's own example request works unchanged:

```bash
curl -s localhost:8420/v1/clairic -H 'content-type: application/json' -d '{
  "model": "clairic",
  "state": "Invoice INV-2291 from Northwind Supplies to Contoso Ltd. Issued September 2, 2026. Payment terms: net 30. Line items: 12 desk lamps at $65.00, 4 monitor arms at $75.00. Total due: $1,080.00. Questions? Contact Dana Ruiz or Priya Shah at billing@northwind.example.",
  "fields": {
    "invoice_number": {"type": "text", "instructions": "The invoice number"},
    "issue_date":     {"type": "date", "instructions": "The date the invoice was issued"},
    "total":          {"type": "number", "instructions": "The total amount due", "decimals": 2},
    "contacts":       {"type": "list", "items": "text", "instructions": "People to contact about the invoice"},
    "po_number":      {"type": "text", "instructions": "The purchase order (PO) number"},
    "payment_terms":  {"type": "duration", "unit": "days", "instructions": "How long the customer has to pay"},
    "email":          {"type": "text", "instructions": "The billing email address"}},
  "reference_date": "2026-10-22"
}'
```

```json
"answers": {
  "invoice_number": {"type": "text", "value": "INV-2291", "p_null": 0.0349, "span": [8, 16], "confidence": 0.3621},
  "issue_date":     {"type": "date", "value": "2026-09-02", "p_null": 0.0326, "span": [64, 81], "confidence": 0.5511},
  "total":          {"type": "number", "value": 1080.0, "p_null": 0.0424, "span": [180, 189], "confidence": 0.7023},
  "contacts":       {"type": "list", "value": ["Dana Ruiz", "Priya Shah"], "p_null": 0.0522, "spans": [[210, 219], [223, 233]], "confidence": 0.1697},
  "po_number":      {"type": "text", "value": null, "p_null": 0.9039, "guess": "INV-2291", "confidence": 0.9039},
  "payment_terms":  {"type": "duration", "value": 30, "p_null": 0.3592, "span": [98, 104], "confidence": 0.0848},
  "email":          {"type": "text", "value": "billing@northwind.example", "p_null": 0.0708, "span": [237, 262], "confidence": 0.1601}}
```

- **`value`** is the typed value. Dates come back as ISO strings, numbers as numbers, durations in `unit`.
- **`value` is `null`** when the state doesn't give it: `p_null` is above `null_threshold` (default 0.5), or nothing of the right type is in the text. `guess` keeps the best candidate anyway. Here the invoice has no PO number, so `po_number` is null with `p_null` 0.90.
- **`span` / `spans`** are character offsets of the value in the state.

How it works: each field becomes a hidden yes/no question ("Does the text give the invoice number?") over the same cached state.
- `p_null` is 1 − P(yes).
- Then every candidate of the field's type in the text (numbers, dates, durations; for text fields, emails, phone numbers, URLs, IDs or names when the instructions name one, else the evidence spans themselves) is scored by the evidence for "yes". The best one wins.

Known limits (beta):
- **Free-text fields can come back cut short.** These are text fields with no type pattern, such as a file name or a version tag, where the value is the evidence span itself. The models tend to leave punctuation inside a value out of the span, so `quarterly.report.final.pdf` can come back as `quarterly.report.final`, or as `quarterly`. Sometimes no value comes back at all. Fields with a type pattern (email, phone, URL, IBAN, IDs, names, addresses, numbers, dates) aren't affected. This is a training limit, and it is on the roadmap.
- **Pattern choice goes by keywords in the instructions,** and the earliest keyword wins. "The file name of the attachment" picks up the person-name pattern because it contains "name"; "The contact email" does too, because "contact" comes before "email". Lead with the kind of value: "the email address to contact", "the attached document".

## Step 5: questions and fields together

```bash
curl -s localhost:8420/v1/horopter -H 'content-type: application/json' -d '{
  "state": "Invoice INV-2291 … Total due: $1,080.00. …",
  "questions": {"has_po": {"type": "noul", "instructions": "Does the invoice reference a purchase order?"}},
  "fields": {"total": {"type": "number", "integer": true}, "issued": {"type": "datetime"}}
}'
```

All answers land in one `answers` object, so question ids and field names must differ.

## Step 6: model names, adapters and strict mode

`GET /v1/models` lists the names. With the published models:

| name | what answers |
|---|---|
| `horopter` | the trunk alone (the default), the adapter given with `--serve`, or the router's pick per question with `--router` |
| `horopter-<adapter>` | one adapter for every question, for each one loaded: `base` (the trunk alone) and, with the flags above, `typed`, `nli`, `relation`, `classification`, `knowledge`, `clause`, `check2` |
| `<any>-strict` | the bare Jev body (`model`, `usage`, `answers` with Jev fields only), byte-compatible with Jev's SDK |
| `mbnli-3-beta*` | the same models under the beta's names; their extension blocks are called `mbnli` |
| `clairic`, `@cf/cloudflare/clairic` | the default model, with the name echoed back |

## Workers AI shape

```bash
curl -s localhost:8420/client/v4/accounts/any/ai/run/@cf/cloudflare/clairic -H 'content-type: application/json' \
  -d '{"model": "@cf/cloudflare/clairic", "state": "…", "fields": {"total": {"type": "number"}}}'
# {"result": {"model": "@cf/cloudflare/clairic", "answers": {"total": {...}}, ...}, "success": true, "errors": [], "messages": []}
```

To point an existing Cloudflare REST client at Horopter, change its base URL. No API token is checked.

## Errors

Errors are a 4xx with `{"detail": ...}`:

| status | when |
|---|---|
| 400 | malformed request (an unknown key, unknown type, missing criteria, bad `reference_date`); a question id equal to a field name; images (Jev `assets`, Clairic `images`) or a `box` question; an `estimate` or `forecast` field |
| 404 | a model name this server doesn't serve (never silently substituted) |
| 413 | the state is longer than the model's window and truncation wasn't allowed |
| 503 | the GPU ran out of memory for this request; the server stays up. Retry, or send fewer questions per request (or start the server with a smaller `--max_batch`) |

## Differences from Jev and Clairic

| | Jev 1.14 | Clairic | Horopter |
|---|---|---|---|
| questions `noul` / `choice` / `score` | yes | no | yes |
| fields `text` / `number` / `date` / `time` / `datetime` / `duration` / `list` | no | yes | yes |
| fields `estimate` / `forecast` | no | yes | **no** (400) |
| dates computed from other values ("due date" = issue date + terms) | n/a | yes | **no**: only dates written in the text |
| images, `box` questions | yes (experimental) | images | **no** (400; reserved) |
| state length | 64k tokens | 65k | **2,048 tokens** (Clairic routes truncate and say so) |
| evidence spans, relation, calibrated abstention | no | spans for text fields | **every answer** |
| runs on your own CPU or GPU | no | no | **yes** |

Notes:
- Field `confidence` is (1 − `p_null`) × the evidence score of the chosen value. It ranks answers but isn't a calibrated probability. `p_null` is the calibrated-looking quantity: E43 measured AUROC 0.95–0.99 for absent vs present values.
- Numeric dates with both parts ≤ 12 are read month first (`02/03/26` → 2026-02-03). Day first is used only when the first number can't be a month.
- A date written without a year takes the year of `reference_date` (default: today).
