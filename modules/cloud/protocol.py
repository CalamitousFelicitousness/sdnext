"""Cloud provider adapter Protocol and result types.

All adapter methods are sync. Result dataclasses carry raw bytes (not file
paths or PIL.Image) so callers can decode for display, V2 can save to disk,
and V1 can base64-encode for HTTP responses without going through any
intermediate format.
"""
# pylint: disable=unnecessary-ellipsis

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


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
