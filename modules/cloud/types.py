from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from PIL import Image as PILImage


@dataclass
class TextRequest:
    model: str
    prompt: str
    system: Optional[str] = None
    prefill: Optional[str] = None
    thinking: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionRequest(TextRequest):
    image: Optional['PILImage.Image'] = None


@dataclass
class TextResponse:
    text: str = ''
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    model: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Any] = None


VisionResponse = TextResponse


@dataclass
class ImageRequest:
    model: str
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    seed: Optional[int] = None
    guidance_scale: Optional[float] = None
    num_images: int = 1
    init_image: Optional['PILImage.Image'] = None
    mask: Optional['PILImage.Image'] = None
    strength: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResponse:
    images: list = field(default_factory=list)
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Any] = None


@dataclass
class VideoRequest:
    model: str
    prompt: str
    duration: Optional[float] = None
    width: int = 1280
    height: int = 720
    fps: Optional[int] = None
    seed: Optional[int] = None
    image: Optional['PILImage.Image'] = None
    num_frames: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoResponse:
    video_bytes: Optional[bytes] = None
    video_path: Optional[str] = None
    frames: list = field(default_factory=list)
    duration: Optional[float] = None
    model: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Any] = None


JobStatus = Literal['pending', 'submitted', 'running', 'succeeded', 'failed', 'cancelled']
TERMINAL_JOB_STATUSES: tuple[str, ...] = ('succeeded', 'failed', 'cancelled')


@dataclass
class Job:
    id: str
    provider_id: str
    capability: str
    status: str = 'pending'
    progress: float = 0.0
    message: str = ''
    result: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    updated_at: float = 0.0
    request: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False


class CloudError(Exception):
    pass
