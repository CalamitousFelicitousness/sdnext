"""Pydantic models for upstream provider responses.

Used by adapter.py to parse provider responses before extracting our internal
result types. Acts as a runtime schema enforcer: if a provider response does
not match the expected shape, parsing raises ValidationError immediately
rather than producing confusing KeyErrors deep in the extraction helpers.

Models are LENIENT (`extra='allow'`) because providers add fields we do not
care about and erroring on those would be brittle. The fields explicitly
declared here are the ones the adapter actually consumes.

Distinct from `modules.cloud.protocol`, which defines our INTERNAL result
types (`ChatResult`, `ImageResult`, etc.) returned by the adapter to callers.
This file is provider-shape; protocol.py is sdnext-shape.
"""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field  # pylint: disable=no-name-in-module


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    type: str | None = None
    function: dict | None = None


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: str
    detail: str | None = None


class ContentPart(BaseModel):
    """A single part of a multimodal chat message content array.

    Different providers emit different shapes for image content:
      - OpenAI:    {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}
      - Some:      {"type": "image", "data": "<base64>"}
      - Some:      {"type": "image", "b64_json": "<base64>"}
    All three shapes are accommodated; extraction selects whichever is present.
    """
    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None
    image_url: ImageURL | None = None
    data: str | None = None
    b64_json: str | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int | None = None
    message: Message
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    """OpenAI-style /v1/chat/completions response shape."""
    model_config = ConfigDict(extra="allow")
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage | None = None


class ImageItem(BaseModel):
    """A single image entry in /v1/images/generations or /v1/images/edits response."""
    model_config = ConfigDict(extra="allow")
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageResponse(BaseModel):
    """OpenAI-style image-generation response shape."""
    model_config = ConfigDict(extra="allow")
    data: list[ImageItem] = Field(default_factory=list)
    usage: Usage | None = None


class VideoSubmitResponse(BaseModel):
    """POST /v1/videos response: provider-assigned job id + initial status.

    Sora returns more fields (model, created_at, expires_at, etc.) but only
    `id` is load-bearing for the poll loop. `status` defaults so providers
    that omit it on submit (returning only the id) still parse cleanly.
    """
    model_config = ConfigDict(extra="allow")
    id: str
    status: str = "queued"


class VideoStatusResponse(BaseModel):
    """GET /v1/videos/{id} response: status + progress + delivery URLs.

    Three competing duration field names show up across providers (Sora uses
    `seconds`, Kling/Pruna use `duration`); both are declared and the consumer
    picks whichever is non-None. Same for the URL fields; see the
    download_video_content precedence in adapter.py.
    """
    model_config = ConfigDict(extra="allow")
    id: str
    status: str
    progress: float | int | None = None
    urls: list[str] | None = None
    unsigned_urls: list[str] | None = None
    video_url: str | None = None
    seconds: float | None = None
    duration: float | None = None
    error: dict | str | None = None
    usage: Usage | None = None


# ---- NanoGPT-specific video shapes -------------------------------------------
#
# NanoGPT's video API differs from Sora's pattern (per their published docs at
# docs.nano-gpt.com/api-reference/video-generation):
#   - submit endpoint:  POST /api/generate-video    (not /v1/videos)
#   - status endpoint:  GET /api/video/status?requestId=...
#   - id field:         `runId` (also returns `id` redundantly)
#   - status enum:      UPPER_CASE (IN_QUEUE / IN_PROGRESS / COMPLETED / FAILED)
#   - video URL:        nested at data.output.video.url
#
# Distinct from VideoSubmitResponse / VideoStatusResponse so each dispatch
# method (generate_video_via_endpoint vs generate_video_via_nanogpt) has its
# own type-correct contract.


class NanogptVideoSubmit(BaseModel):
    """POST /api/generate-video response.

    Returns both `runId` and `id`; we accept either via field alias so
    downstream code reads `submit.id` regardless of which key the provider
    populated. `cost` and `remainingBalance` get extra='allow'-captured for
    debugging.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str = Field(validation_alias=AliasChoices("id", "runId"))
    status: str = "pending"


class NanogptVideoOutput(BaseModel):
    """Nested data.output payload. `video` is the dict containing `url`,
    `videoUrls` is the redundant array form some Pruna routes return."""
    model_config = ConfigDict(extra="allow")
    video: dict | None = None
    videoUrls: list[str] | None = None


class NanogptVideoData(BaseModel):
    """Nested `data` envelope. NanoGPT puts status, request_id, progress,
    duration, error, and the output URL all inside `data` (the docs were
    misleading; this matches the actual response from /api/video/status)."""
    model_config = ConfigDict(extra="allow")
    status: str | None = None
    request_id: str | None = None
    progress: float | int | None = None
    duration: float | None = None
    error: dict | str | None = None
    output: NanogptVideoOutput | None = None


class NanogptVideoStatus(BaseModel):
    """GET /video/status?requestId=... response.

    Top-level wraps {requestId, model, data}. The interesting fields all
    live inside `data`. Status values use UPPER_CASE: IN_QUEUE / IN_PROGRESS
    / COMPLETED / FAILED. Adapter accesses status via `parsed.data.status`.
    """
    model_config = ConfigDict(extra="allow")
    requestId: str | None = None
    model: str | None = None
    data: NanogptVideoData | None = None
    usage: Usage | None = None
