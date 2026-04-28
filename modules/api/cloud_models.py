from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class CloudProviderInfo(BaseModel):
    id: str = Field(..., description="Stable provider identifier (e.g. 'openai', 'anthropic', 'openrouter')")
    label: str = Field(..., description="Human-readable provider label")
    capabilities: list[str] = Field(default_factory=list, description="List of capabilities the provider supports: 'text', 'vision'")
    enabled: bool = Field(False, description="Whether the provider has valid credentials configured")
    models: list[str] = Field(default_factory=list, description="Default model identifiers offered by this provider")


class CloudModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier as accepted by the provider's API")
    label: Optional[str] = Field(None, description="Optional human-readable label")
    supports_vision: bool = Field(False, description="Whether the model accepts image inputs")
    supports_streaming: bool = Field(False, description="Whether the model supports SSE streaming via this framework")
    supports_image: bool = Field(False, description="Whether the model supports image generation")
    supports_video: bool = Field(False, description="Whether the model supports video generation")


class CloudTextRequest(BaseModel):
    provider: str = Field(..., description="Provider id from /sdapi/v1/cloud/providers")
    model: str = Field(..., description="Model id (provider-prefix optional; will be stripped if matching)")
    prompt: str = Field(..., description="User prompt")
    system: Optional[str] = Field(None, description="System prompt")
    prefill: Optional[str] = Field(None, description="Assistant prefill text appended to the user message for OpenAI-compat providers, sent as an assistant message for Anthropic")
    thinking: bool = Field(False, description="Enable provider-specific thinking/reasoning mode if supported")
    temperature: Optional[float] = Field(None)
    max_tokens: Optional[int] = Field(None)
    top_p: Optional[float] = Field(None)
    top_k: Optional[int] = Field(None)
    stream: bool = Field(False, description="If true and called against POST /cloud/text/stream, returns SSE chunks")
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific passthrough fields")


class CloudVisionRequest(CloudTextRequest):
    image: str = Field(..., description="Image as base64 string or data URL (e.g. 'data:image/jpeg;base64,...')")


class CloudTextResponse(BaseModel):
    text: str = Field("", description="Generated text content")
    finish_reason: Optional[str] = Field(None)
    usage: Optional[dict] = Field(None, description="Provider-specific usage stats (input_tokens/output_tokens or similar)")
    model: Optional[str] = Field(None, description="Actual model that responded")
    error: Optional[str] = Field(None, description="Error message if the call failed; text will be empty in that case")


class CloudImageRequest(BaseModel):
    provider: str = Field(..., description="Image provider id from /sdapi/v1/cloud/providers")
    model: str = Field(..., description="Model id (provider-prefix optional; will be stripped if matching)")
    prompt: str = Field(..., description="User prompt")
    negative_prompt: Optional[str] = Field(None)
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)
    steps: int = Field(30, ge=1, le=200)
    seed: Optional[int] = Field(None)
    guidance_scale: Optional[float] = Field(None)
    num_images: int = Field(1, ge=1, le=8)
    init_image: Optional[str] = Field(None, description="img2img: base64 string or data URL")
    mask: Optional[str] = Field(None, description="inpaint: base64 string or data URL")
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific passthrough fields")


class CloudVideoRequest(BaseModel):
    provider: str = Field(..., description="Video provider id from /sdapi/v1/cloud/providers")
    model: str = Field(..., description="Model id (provider-prefix optional; will be stripped if matching)")
    prompt: str = Field(..., description="User prompt")
    duration: Optional[float] = Field(None, description="Target duration in seconds (provider-specific interpretation)")
    width: int = Field(1280, ge=64, le=4096)
    height: int = Field(720, ge=64, le=4096)
    fps: Optional[int] = Field(None, ge=1, le=120)
    seed: Optional[int] = Field(None)
    image: Optional[str] = Field(None, description="I2V conditioning image: base64 string or data URL")
    num_frames: Optional[int] = Field(None, ge=1, description="Frame count if duration is not specified (Veo back-compat)")
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific passthrough fields")


class CloudImageResultInline(BaseModel):
    images: list[str] = Field(default_factory=list, description="Generated images as data URLs (data:image/png;base64,...)")
    finish_reason: Optional[str] = Field(None)
    model: Optional[str] = Field(None)


class CloudVideoResultInline(BaseModel):
    video_b64: Optional[str] = Field(None, description="Generated video as base64 string (mp4)")
    video_path: Optional[str] = Field(None, description="Optional server-side filesystem path if the provider wrote a file")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    model: Optional[str] = Field(None)


class CloudJob(BaseModel):
    id: str = Field(..., description="Job id (uuid4 hex)")
    provider_id: str = Field(...)
    capability: str = Field(..., description="One of 'image' | 'video'")
    status: str = Field(..., description="One of 'pending' | 'submitted' | 'running' | 'succeeded' | 'failed' | 'cancelled'")
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = Field("")
    error: Optional[str] = Field(None)
    started_at: float = Field(0.0)
    updated_at: float = Field(0.0)
    result: Optional[Any] = Field(None, description="CloudImageResultInline | CloudVideoResultInline (typed by capability)")
