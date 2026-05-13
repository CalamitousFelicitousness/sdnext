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

from pydantic import BaseModel, ConfigDict, Field  # pylint: disable=no-name-in-module


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
