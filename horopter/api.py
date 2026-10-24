# SPDX-License-Identifier: MIT
"""Wire format: one request shape that is a superset of Jev's System One (typed questions over a state) and of
Cloudflare's Clairic (typed fields extracted from a state), and one response shape that is a superset of both answers.

- A Jev client posts to /v1/systemone and reads model / usage / answers; the extra keys are ignored by its SDK.
- A Clairic client posts to /v1/clairic (or Workers AI's /ai/run/@cf/cloudflare/clairic, wrapped in `result`) and reads
  model / answers / truncated / usage / state_text.
- /v1/horopter is the same handler under our own name: questions and fields in one request, over one cached state.

Question ids and field names share one `answers` object, so they must not collide. Unknown keys are refused (400,
naming the key), never silently dropped. The extension block (relation,
abstention, evidence, version) is `horopter` (or `mbnli` for the mbnli-3-beta* model names); `-strict` model names drop
it and every other non-Jev key.
"""
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Text = Union[str, dict, list]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Closed(BaseModel):
    """Request objects: an unknown key is a 400 naming it (a misspelt option must not be dropped without a word)."""
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------------------------- request
class NoulQuestion(_Closed):
    type: Literal["noul"]
    instructions: Optional[Text] = Field(None, description="The yes/no question. Missing: the question id is used.")
    criteria: Optional[dict[str, Text]] = Field(None, description="Optional descriptions of `true` and `false`.")


class ChoiceQuestion(_Closed):
    type: Literal["choice"]
    instructions: Optional[Text] = None
    criteria: dict[str, Text] = Field(..., description="Option name -> description; at least 2.")


class ScoreQuestion(_Closed):
    type: Literal["score"]
    instructions: Optional[Text] = None
    criteria: list[Text] = Field(..., description="Ordered levels, lowest first; at least 2.")


class BoxQuestion(_Closed):
    type: Literal["box"]
    instructions: Optional[Text] = Field(None, description="Jev 1.14: locate something in an image. Reserved: this "
                                                           "model has no image input yet, and refuses box questions.")


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, BoxQuestion], Field(discriminator="type")]

FieldType = Literal["text", "number", "date", "time", "datetime", "duration", "list", "estimate", "forecast"]


class FieldSpec(_Closed):
    type: FieldType
    instructions: Optional[str] = Field(None, description="What the field should contain.")
    unit: Optional[str] = Field(None, description="Durations: seconds, minutes, hours, days, weeks, months, years.")
    integer: Optional[bool] = None
    decimals: Optional[int] = Field(None, ge=0, le=6)
    items: Optional[Literal["text", "number", "date"]] = Field(None, description="Required for list fields.")
    min: Optional[Any] = None
    max: Optional[Any] = None
    step: Optional[Any] = None
    scale: Optional[Literal["linear", "log"]] = None
    null_threshold: Optional[float] = Field(None, gt=0, le=1, description="p_null above which the value is null "
                                                                           "(default 0.5).")
    horizon: Optional[int] = Field(None, ge=1, le=64)
    interval: Optional[str] = None


class Asset(_Closed):
    """Jev 1.14 image asset, referenced by its key in backticks from the state or instructions."""
    format: Literal["png", "jpg", "jpeg"]
    data_base64: str
    type: Optional[Literal["image"]] = None


class Base64Image(_Closed):
    content_type: str
    base64: str


class Options(_Closed):
    include_quantiles: bool = False
    return_state_text: bool = False
    allow_truncation: Optional[bool] = Field(None, description="Read the first window of a too-long state instead of "
                                                               "refusing (default: true on Clairic routes, else false).")


class Extensions(_Closed):
    """Horopter request options (also accepted under the key `mbnli`)."""
    evidence: Optional[dict[str, Any]] = Field(None, description='{"mode": "off" | "chosen" | "all", "threshold": '
                                                                 '0..1}: evidence spans per question answer.')
    allow_truncation: Optional[bool] = None
    debug: Optional[bool] = Field(None, description="Add routing and hidden-question details.")


class DecideRequest(_Closed):
    model: Optional[str] = Field(None, description="A served model name (GET /v1/models); `clairic` maps to the "
                                                   "default model.")
    state: Union[str, dict, list] = Field("", description="Text, or JSON (rendered with sorted keys).")
    questions: Optional[dict[str, Question]] = Field(None, description="Jev typed questions, by id.")
    fields: Optional[dict[str, FieldSpec]] = Field(None, description="Clairic typed fields, by name.")
    assets: Optional[dict[str, Asset]] = Field(None, description="Jev image assets (refused by this model).")
    images: Optional[list[Union[str, Base64Image]]] = Field(None, description="Clairic images (refused by this "
                                                                              "model).")
    reference_date: Optional[str] = Field(None, description="ISO date anchoring dates written without a year.")
    options: Optional[Options] = None
    horopter: Optional[Extensions] = None
    mbnli: Optional[Extensions] = None
    evidence: Optional[Union[Literal["off", "chosen", "all"], dict[str, Any]]] = Field(
        None, description='Shorthand for horopter.evidence: a mode ("chosen") or the full object.')


class TokenizeRequest(_Closed):
    model: Optional[str] = Field(None, description="A served model name (all share one tokenizer); checked if given.")
    state: Union[str, dict, list] = Field("", description="Text, or JSON (rendered with sorted keys), as in a request.")


# ------------------------------------------------------------------------------------------------- response (docs)
class Usage(_Open):
    input_tokens: int
    output_tokens: int
    state_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
    schema_tokens: Optional[int] = None
    passes: Optional[int] = None


class Answer(_Open):
    """One answer. Questions: Jev fields (noul | choice, confidence, probabilities | score, confidence, legend,
    probabilities). Fields: Clairic fields (value, guess, p_null, confidence, span | spans)."""
    type: str
    noul: Optional[float] = None
    choice: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict[str, float]] = None
    legend: Optional[dict[str, Any]] = None
    value: Optional[Any] = None
    guess: Optional[Any] = None
    p_null: Optional[float] = None
    span: Optional[list[int]] = None
    spans: Optional[list[list[int]]] = None
    horopter: Optional[dict[str, Any]] = Field(None, description="relation, abstain, evidence (mbnli on mbnli-* names)")


class DecideResponse(_Open):
    model: str
    usage: Usage
    answers: dict[str, Answer]
    truncated: Optional[bool] = None
    state_text: Optional[str] = None
    horopter: Optional[dict[str, Any]] = Field(None, description="version, state_id, timings (mbnli on mbnli-* names)")


class TokenizeResponse(_Open):
    state_text: str = Field(..., description="The rendered state: the text that span offsets index into.")
    state_tokens: int = Field(..., description="Tokens in the whole rendered state.")
    limit: int = Field(..., description="Tokens this model reads; a longer state is a 413 (or truncated on request).")
    truncated: bool = Field(..., description="state_tokens > limit.")
    read_chars: int = Field(..., description="Characters of state_text inside the window (all of it if not truncated).")
    offsets: list[list[int]] = Field(..., description="[start, end) character offsets of each token in state_text.")
    state_id: str = Field(..., description="The id decide() reports for this state (sha256 of state_text).")


class WorkersAIResponse(_Open):
    result: DecideResponse
    success: bool
    errors: list
    messages: list
