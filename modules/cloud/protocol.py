"""Cloud provider adapter Protocol and result types.

All adapter methods are sync. Result dataclasses carry raw bytes (not file
paths or PIL.Image) so callers can decode for display, V2 can save to disk,
and V1 can base64-encode for HTTP responses without going through any
intermediate format.

Pydantic SizeConstraint variants (discriminated union) sourced from
modules/cloud/size_constraints.json represent the internal size-constraint
shape; provider-shape parsing lives in response_models.py.
"""
# pylint: disable=unnecessary-ellipsis

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator  # pylint: disable=no-name-in-module


@dataclass
class CloudUsage:
    """Token / cost reporting from a cloud call. Embedded in result dataclasses.

    All fields optional because providers like Ollama and self-hosted endpoints
    do not report usage.
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


@dataclass
class ChatResult:
    """Result of a cloud chat / VLM call. Used by the text functions."""
    content: str = ""
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"
    usage: CloudUsage | None = None


@dataclass
class ImageResult:
    """Result of a cloud image generation."""
    images: list[bytes] = field(default_factory=list)
    revised_prompt: str | None = None
    format: str = "png"
    usage: CloudUsage | None = None


# ---- image size constraints --------------------------------------------------
#
# Discriminated union expressing the shape of a model's output-size domain.
# Loaded from modules/cloud/size_constraints.json by adapter.py, surfaced on
# normalize_models() output, consumed by the pre-flight validation hook in
# image.py. Each variant is independently constructable; the discriminator on
# `kind` lets Pydantic dispatch to the right class via TypeAdapter.


class SizeConstraintBase(BaseModel):
    """Shared fields across all size_constraint variants.

    `extra="forbid"` because this is sdnext-shape data we control; unexpected
    keys signal a stale or corrupt size_constraints.json that should fail loud.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    allow_auto: bool = False
    auto_wire: Literal["literal", "omit", "default"] | None = None
    default: str | None = None

    @model_validator(mode="after")
    def validate_auto_wire_only_when_allowed(self):
        if self.auto_wire is not None and not self.allow_auto:
            raise ValueError("auto_wire is meaningful only when allow_auto=True")
        return self


class SizeConstraintEnum(SizeConstraintBase):
    """Discrete WxH preset list. `options` never includes the literal 'auto'
    string; auto support is signalled by `allow_auto`."""

    kind: Literal["enum"] = "enum"
    options: list[str]


class SizeConstraintBucket(SizeConstraintBase):
    """Symbolic-label sizing (e.g. '1k' / '2k' / '4k'). The server resolves the
    label to concrete pixel dimensions, so wire requests carry the symbol.
    `resolve` is REQUIRED documentation so UIs can render
    'pseudo-resolution: ~2048x2048' alongside each bucket label."""

    kind: Literal["bucket"] = "bucket"
    options: list[str]
    resolve: dict[str, dict[str, int]]


class SizeConstraintFree(SizeConstraintBase):
    """Continuous WxH with one or more bounds. Validation checks every populated
    bound; absent bounds are treated as unconstrained. `align` may be a single
    int (both axes share an alignment) or (width_align, height_align)."""

    kind: Literal["free"] = "free"
    min_pixel_count: int | None = None
    max_pixel_count: int | None = None
    min_longest_side: int | None = None
    max_longest_side: int | None = None
    aspect_ratio_min: float | None = None
    aspect_ratio_max: float | None = None
    align: int | tuple[int, int] | None = None


SizeConstraint = Annotated[
    SizeConstraintEnum | SizeConstraintBucket | SizeConstraintFree,
    Field(discriminator="kind"),
]


@dataclass
class AudioResult:
    """Result of a cloud TTS call."""
    data: bytes = b""
    format: str = "mp3"
    duration: float | None = None


@dataclass
class TranscribeResult:
    """Result of a cloud STT call."""
    text: str = ""
    language: str | None = None
    segments: list[dict] | None = None
    duration: float | None = None


@dataclass
class VideoResult:
    """Result of a cloud video generation."""
    data: bytes = b""
    format: str = "mp4"
    duration: float | None = None
    thumbnail: bytes | None = None
    usage: CloudUsage | None = None


# Progress callback shape: dict with at minimum a "phase" key.
# Standard phases: "submitted", "queued_remote", "processing", "downloading".
# Adapters may add provider-specific fields freely.
ProgressCallback = Callable[[dict], None]


class ProviderAdapter(Protocol):
    """Sync adapter contract.

    Text implementations live in chat, list_models, and validate_key; image
    generation lives in generate_image; video and audio methods raise
    NotImplementedError until those code paths land. The Protocol surface
    is locked so the V2 layer can type-import it without revision.
    """

    def list_models(self) -> list[dict]: ...
    def chat(self, params: dict, on_progress: ProgressCallback) -> ChatResult: ...
    def generate_image(self, params: dict, on_progress: ProgressCallback) -> ImageResult: ...
    def tts(self, params: dict) -> AudioResult: ...
    def transcribe(self, params: dict) -> TranscribeResult: ...
    def generate_video(self, params: dict, on_progress: ProgressCallback) -> VideoResult: ...
    def cancel(self, remote_id: str) -> bool: ...
    def probe_endpoints(self) -> dict[str, bool]: ...
    def validate_key(self) -> bool: ...
