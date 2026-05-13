"""V1 REST API for cloud providers and text endpoints.

Routes registered via register_api(api). Auth is automatic via
Api.add_api_route(auth=True).

Error envelope is {error, kind, ...}. CloudError subclasses are mapped via a
global exception handler so route handlers stay focused on the happy path.
Local validation failures (e.g. bad base64) return JSONResponse directly
rather than raising HTTPException, so they match the same envelope shape.
"""

import base64
import json

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

from modules.cloud import image, registry, text, video
from modules.cloud.errors import (
    AuthError,
    CloudError,
    ContentFilterError,
    InputValidationError,
    ModelNotFoundError,
    ProviderError,
    QuotaError,
    RateLimitError,
)


# ---- request/response models -----------------------------------------------------


class ItemProvider(BaseModel):
    id: str = Field(title="Provider id", description="Stable, snake_case provider id derived from name at creation.")
    name: str = Field(title="Provider name")
    preset: str = Field(title="Preset", description="One of: openai, openrouter, nanogpt, aihubmix, ollama, custom.")
    base_url: str = Field(title="Base URL")
    enabled: bool = Field(title="Enabled")
    has_key: bool = Field(title="Has API key", description="True if a key is configured (env var or stored secret); the key value is never returned.")


class ResProviders(BaseModel):
    providers: list[ItemProvider]


class ReqProviderCreate(BaseModel):
    name: str = Field(title="Display name")
    preset: str = Field(title="Preset", description="One of: openai, openrouter, nanogpt, aihubmix, ollama, custom.")
    base_url: str = Field(title="Base URL")
    key: str = Field(default="", title="API key", description="Stored in secrets.json via the cloud_<id>_key option.")


class ResProviderCreate(ItemProvider):
    valid: bool | None = Field(default=None, title="Key valid", description="Result of automatic post-create validation. Null if no key supplied.")
    error: str | None = Field(default=None, title="Validation error")


class ReqProviderUpdate(BaseModel):
    name: str | None = Field(default=None, title="Display name")
    base_url: str | None = Field(default=None, title="Base URL")
    key: str | None = Field(default=None, title="API key")
    enabled: bool | None = Field(default=None, title="Enabled")


class ResProviderDelete(BaseModel):
    deleted: bool


class ResProviderValidate(BaseModel):
    valid: bool
    error: str | None = None


class ResProviderRefresh(BaseModel):
    model_count: int


class ResProviderModels(BaseModel):
    models: list[dict]
    total: int


class ReqPromptEnhance(BaseModel):
    prompt: str = Field(title="Prompt", description="The base prompt to enhance.")
    provider: str = Field(title="Provider id")
    model: str = Field(title="Model id")
    system_prompt: str = Field(default="", title="System prompt override", description="Optional. Empty falls back to a default sensitive to the nsfw flag.")
    nsfw: bool = Field(default=True, title="Allow NSFW", description="Whether the default system prompt permits NSFW content. Ignored if system_prompt is supplied.")


class ResPromptEnhance(BaseModel):
    enhanced: str
    provider: str
    model: str


class ReqCaption(BaseModel):
    image: str = Field(title="Image", description="Base64-encoded PNG / JPEG / WEBP bytes.")
    provider: str = Field(title="Provider id")
    model: str = Field(title="Model id", description="Must be a vision-capable model on the chosen provider.")
    prompt: str = Field(default="Describe this image in detail.", title="Caption prompt")


class ResCaption(BaseModel):
    caption: str
    provider: str
    model: str


class ReqVQA(BaseModel):
    image: str = Field(title="Image", description="Base64-encoded PNG / JPEG / WEBP bytes.")
    question: str = Field(title="Question")
    provider: str = Field(title="Provider id")
    model: str = Field(title="Model id", description="Must be a vision-capable model on the chosen provider.")


class ResVQA(BaseModel):
    answer: str
    provider: str
    model: str


class ReqCloudTxt2Img(BaseModel):
    prompt: str = Field(title="Prompt")
    provider: str = Field(title="Provider id")
    model: str = Field(title="Model id", description="Image-capable model on the chosen provider.")
    negative_prompt: str = Field(default="", title="Negative prompt")
    width: int = Field(default=1024, title="Width")
    height: int = Field(default=1024, title="Height")
    n: int = Field(default=1, title="Image count", description="Number of images to generate. Provider may cap or ignore (e.g. dall-e-3 only supports n=1).")
    seed: int = Field(default=-1, title="Seed", description="-1 picks a random seed at the orchestrator layer; the same seed is passed to the provider.")
    steps: int = Field(default=28, title="Steps", description="Inference steps. Mapped via preset's image param map.")
    guidance_scale: float = Field(default=7.5, title="Guidance scale")
    quality: str = Field(default="standard", title="Quality", description="Provider-specific. 'standard' or 'hd' for DALL-E 3.")
    style: str | None = Field(default=None, title="Style", description="Provider-specific. 'vivid' or 'natural' for DALL-E 3.")
    extra_params: dict = Field(default_factory=dict, title="Extra params", description="Provider-specific passthrough merged into the request body last.")
    save_images: bool = Field(default=True, title="Save to disk", description="Defaults to True (cloud images cost money).")
    send_images: bool = Field(default=True, title="Return base64 in response")


class ReqCloudImg2Img(ReqCloudTxt2Img):
    init_image: str = Field(title="Init image", description="Base64-encoded PNG / JPEG / WEBP. Cloud APIs accept one image only - use the first init image.")
    mask: str | None = Field(default=None, title="Mask", description="Base64-encoded mask. Caller convention: white=editable on black. OpenAI presets invert internally.")
    strength: float = Field(default=0.75, title="Denoise strength")


class ResCloudImage(BaseModel):
    images: list[str] = Field(default_factory=list, title="Generated images", description="Base64 when send_images=True; empty otherwise.")
    saved_paths: list[str] = Field(default_factory=list, title="Saved paths", description="Disk paths when save_images=True; empty otherwise.")
    revised_prompt: str | None = Field(default=None, title="Revised prompt", description="Provider-revised prompt (e.g. dall-e-3 expands prompts).")
    provider: str
    model: str
    info: str = Field(description="JSON-encoded generation metadata. PNG `parameters` text chunk also embeds an sdnext-style infotext.")
    parameters: dict = Field(description="Request echoed back.")
    usage: dict | None = Field(default=None, title="Usage", description="Token / cost reporting if the provider returned usage.")


class ReqCloudVideo(BaseModel):
    prompt: str = Field(title="Prompt")
    provider: str = Field(title="Provider id")
    model: str = Field(title="Model id", description="Video-capable model on the chosen provider (e.g. pruna-ai/p-video/text-to-video, sora-2, kling-v26-pro).")
    aspect_ratio: str | None = Field(default=None, title="Aspect ratio", description="Provider-specific (e.g. '16:9', '9:16', '1:1'). Sora uses orientation instead.")
    duration: float | None = Field(default=None, title="Duration (seconds)", description="Provider-specific clamps apply (typically 1-60).")
    size: str | None = Field(default=None, title="Size", description="Pixel dimensions like '1280x720'. Some providers prefer aspect_ratio over size.")
    init_image: str | None = Field(default=None, title="Init image", description="Base64-encoded PNG / JPEG / WEBP. Presence triggers image-to-video (i2v).")
    seed: int = Field(default=-1, title="Seed", description="-1 picks a random seed at the orchestrator layer; the same seed is passed to the provider.")
    extra_params: dict = Field(default_factory=dict, title="Extra params", description="Provider-specific passthrough merged into the request body last. Use this for Sora's `seconds`/`orientation`/`resolution` fields.")
    save_video: bool = Field(default=True, title="Save to disk", description="Defaults to True (cloud videos cost money).")
    send_video: bool = Field(default=True, title="Return base64 in response")


class ResCloudVideo(BaseModel):
    video: str | None = Field(default=None, title="Generated video", description="Base64-encoded video bytes when send_video=True; null otherwise.")
    saved_path: str | None = Field(default=None, title="Saved path", description="Disk path when save_video=True; null otherwise.")
    thumbnail: str | None = Field(default=None, title="Thumbnail", description="Base64-encoded PNG of the first frame; null on extraction failure.")
    duration: float | None = Field(default=None, title="Duration (seconds)", description="Provider-reported duration if available.")
    format: str = Field(default="mp4", title="Format", description="Video container format (mp4, webm, etc.).")
    provider: str
    model: str
    info: str = Field(description="JSON-encoded generation metadata.")
    parameters: dict = Field(description="Request echoed back.")
    usage: dict | None = Field(default=None, title="Usage", description="Token / cost reporting if the provider returned usage.")


# ---- error mapping ---------------------------------------------------------------


def kind_from_error(exc: CloudError) -> str:
    if isinstance(exc, AuthError):
        return "auth"
    if isinstance(exc, QuotaError):
        return "quota"
    if isinstance(exc, ContentFilterError):
        return "content_filter"
    if isinstance(exc, ModelNotFoundError):
        return "model_not_found"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, InputValidationError):
        return "input_validation"
    return "provider"


def http_status_from_error(exc: CloudError) -> int:
    # InputValidationError is a caller-input failure (pre-upload); 400.
    if isinstance(exc, InputValidationError):
        return 400
    # ProviderError carries the upstream 5xx code; surface as 502 Bad Gateway
    # since the failure was upstream rather than in this server.
    if isinstance(exc, ProviderError):
        return 502
    return exc.status or 500


def cloud_error_response(exc: CloudError) -> JSONResponse:
    body: dict = {"error": str(exc), "kind": kind_from_error(exc)}
    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        body["retry_after"] = exc.retry_after
    if isinstance(exc, ProviderError) and exc.status:
        body["status"] = exc.status
    if isinstance(exc, InputValidationError):
        if exc.field is not None:
            body["field"] = exc.field
        if exc.limit is not None:
            body["limit"] = exc.limit
    if exc.provider:
        body["provider"] = exc.provider
    return JSONResponse(status_code=http_status_from_error(exc), content=body)


def not_found_response(provider_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": f"Provider not found: {provider_id}", "kind": "model_not_found"},
    )


def bad_request_response(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": message, "kind": "input_validation"})


# ---- helpers ---------------------------------------------------------------------


def provider_to_item(cfg) -> ItemProvider:
    has_key = bool(registry.resolve_key(cfg.id, cfg.preset))
    return ItemProvider(
        id=cfg.id,
        name=cfg.name,
        preset=cfg.preset,
        base_url=cfg.base_url,
        enabled=cfg.enabled,
        has_key=has_key,
    )


def decode_image(b64: str):
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as e:
        return None, f"Invalid base64 image: {e}"


# ---- route handlers --------------------------------------------------------------


def get_providers():
    return ResProviders(providers=[provider_to_item(p) for p in registry.list_providers()])


def post_providers(req: ReqProviderCreate):
    cfg = registry.add_provider(req.name, req.preset, req.base_url, req.key)
    valid: bool | None = None
    error: str | None = None
    if req.key:
        valid, error = registry.validate_provider(cfg.id)
    item = provider_to_item(cfg)
    return ResProviderCreate(
        id=item.id,
        name=item.name,
        preset=item.preset,
        base_url=item.base_url,
        enabled=item.enabled,
        has_key=item.has_key,
        valid=valid,
        error=error,
    )


def put_provider(provider_id: str, req: ReqProviderUpdate):
    kwargs = {k: v for k, v in req.model_dump(exclude_unset=True).items() if v is not None}
    cfg = registry.update_provider(provider_id, **kwargs)
    if cfg is None:
        return not_found_response(provider_id)
    return provider_to_item(cfg)


def delete_provider(provider_id: str):
    deleted = registry.remove_provider(provider_id)
    if not deleted:
        return not_found_response(provider_id)
    return ResProviderDelete(deleted=True)


def post_validate(provider_id: str):
    if registry.get_provider(provider_id) is None:
        return not_found_response(provider_id)
    valid, error = registry.validate_provider(provider_id)
    return ResProviderValidate(valid=valid, error=error)


def post_refresh(provider_id: str):
    if registry.get_provider(provider_id) is None:
        return not_found_response(provider_id)
    models = registry.refresh_models(provider_id)
    return ResProviderRefresh(model_count=len(models))


def get_models(provider_id: str):
    if registry.get_provider(provider_id) is None:
        return not_found_response(provider_id)
    adapter = registry.get_adapter(provider_id)
    models = adapter.list_models()
    return ResProviderModels(models=models, total=len(models))


def post_prompt_enhance(req: ReqPromptEnhance):
    if not req.prompt or not req.prompt.strip():
        return bad_request_response("prompt is required")
    enhanced = text.enhance_prompt(
        req.prompt,
        req.provider,
        req.model,
        system_prompt=req.system_prompt or "",
        nsfw=req.nsfw,
    )
    return ResPromptEnhance(enhanced=enhanced, provider=req.provider, model=req.model)


def post_caption(req: ReqCaption):
    try:
        image_bytes = base64.b64decode(req.image, validate=True)
    except Exception as e:
        return bad_request_response(f"Invalid base64 image: {e}")
    if not image_bytes:
        return bad_request_response("image is empty")
    caption_text = text.caption(image_bytes, req.provider, req.model, prompt=req.prompt)
    return ResCaption(caption=caption_text, provider=req.provider, model=req.model)


def post_vqa(req: ReqVQA):
    try:
        image_bytes = base64.b64decode(req.image, validate=True)
    except Exception as e:
        return bad_request_response(f"Invalid base64 image: {e}")
    if not image_bytes:
        return bad_request_response("image is empty")
    if not req.question or not req.question.strip():
        return bad_request_response("question is required")
    answer = text.vqa(image_bytes, req.question, req.provider, req.model)
    return ResVQA(answer=answer, provider=req.provider, model=req.model)


def call_cloud_image(req: ReqCloudTxt2Img, init_bytes: bytes | None = None, mask_bytes: bytes | None = None) -> ResCloudImage:
    """Shared body for the txt2img and img2img handlers."""
    result = image.generate_image(
        prompt=req.prompt,
        provider_id=req.provider,
        model=req.model,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        n=req.n,
        seed=req.seed,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        quality=req.quality,
        style=req.style,
        init_image=init_bytes,
        mask=mask_bytes,
        strength=getattr(req, "strength", 0.75),
        extra_params=req.extra_params or None,
        save_to_disk=req.save_images,
    )
    images_b64: list[str] = []
    if req.send_images:
        images_b64 = [base64.b64encode(b).decode("ascii") for b in result.images]
    return ResCloudImage(
        images=images_b64,
        saved_paths=result.saved_paths,
        revised_prompt=result.revised_prompt,
        provider=result.provider,
        model=result.model,
        info=json.dumps(result.info),
        parameters=req.model_dump(),
        usage=result.info.get("usage"),
    )


def post_cloud_txt2img(req: ReqCloudTxt2Img):
    if not req.prompt or not req.prompt.strip():
        return bad_request_response("prompt is required")
    return call_cloud_image(req)


def post_cloud_img2img(req: ReqCloudImg2Img):
    if not req.prompt or not req.prompt.strip():
        return bad_request_response("prompt is required")
    if not req.init_image:
        return bad_request_response("init_image is required for img2img")
    try:
        init_bytes = base64.b64decode(req.init_image, validate=True)
    except Exception as e:
        return bad_request_response(f"Invalid base64 init_image: {e}")
    if not init_bytes:
        return bad_request_response("init_image is empty")
    mask_bytes: bytes | None = None
    if req.mask:
        try:
            mask_bytes = base64.b64decode(req.mask, validate=True)
        except Exception as e:
            return bad_request_response(f"Invalid base64 mask: {e}")
        if not mask_bytes:
            return bad_request_response("mask is empty")
    return call_cloud_image(req, init_bytes=init_bytes, mask_bytes=mask_bytes)


def post_cloud_video(req: ReqCloudVideo):
    """Single endpoint for both t2v and i2v; init_image presence dispatches."""
    if not req.prompt or not req.prompt.strip():
        return bad_request_response("prompt is required")
    init_bytes: bytes | None = None
    if req.init_image:
        try:
            init_bytes = base64.b64decode(req.init_image, validate=True)
        except Exception as e:
            return bad_request_response(f"Invalid base64 init_image: {e}")
        if not init_bytes:
            return bad_request_response("init_image is empty")
    result = video.generate_video(
        prompt=req.prompt,
        provider_id=req.provider,
        model=req.model,
        aspect_ratio=req.aspect_ratio,
        duration=req.duration,
        size=req.size,
        init_image=init_bytes,
        seed=req.seed,
        extra_params=req.extra_params or None,
        save_to_disk=req.save_video,
    )
    video_b64: str | None = None
    if req.send_video and result.video:
        video_b64 = base64.b64encode(result.video).decode("ascii")
    thumbnail_b64: str | None = None
    if result.thumbnail:
        thumbnail_b64 = base64.b64encode(result.thumbnail).decode("ascii")
    return ResCloudVideo(
        video=video_b64,
        saved_path=result.saved_path,
        thumbnail=thumbnail_b64,
        duration=result.duration,
        format=result.format,
        provider=result.provider,
        model=result.model,
        info=json.dumps(result.info),
        parameters=req.model_dump(),
        usage=result.info.get("usage"),
    )


# ---- registration ----------------------------------------------------------------


def register_api(api):
    """Register cloud V1 routes onto the sdnext Api instance."""

    async def cloud_error_handler(_request: Request, exc: CloudError):
        return cloud_error_response(exc)

    api.app.add_exception_handler(CloudError, cloud_error_handler)

    api.add_api_route("/sdapi/v1/cloud/providers", get_providers, methods=["GET"], response_model=ResProviders, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/providers", post_providers, methods=["POST"], response_model=ResProviderCreate, status_code=201, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/providers/{provider_id}", put_provider, methods=["PUT"], response_model=ItemProvider, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/providers/{provider_id}", delete_provider, methods=["DELETE"], response_model=ResProviderDelete, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/providers/{provider_id}/validate", post_validate, methods=["POST"], response_model=ResProviderValidate, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/providers/{provider_id}/refresh", post_refresh, methods=["POST"], response_model=ResProviderRefresh, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/providers/{provider_id}/models", get_models, methods=["GET"], response_model=ResProviderModels, tags=["Cloud"])

    api.add_api_route("/sdapi/v1/cloud/prompt-enhance", post_prompt_enhance, methods=["POST"], response_model=ResPromptEnhance, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/caption", post_caption, methods=["POST"], response_model=ResCaption, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/vqa", post_vqa, methods=["POST"], response_model=ResVQA, tags=["Cloud"])

    api.add_api_route("/sdapi/v1/cloud/txt2img", post_cloud_txt2img, methods=["POST"], response_model=ResCloudImage, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/img2img", post_cloud_img2img, methods=["POST"], response_model=ResCloudImage, tags=["Cloud"])

    api.add_api_route("/sdapi/v1/cloud/video", post_cloud_video, methods=["POST"], response_model=ResCloudVideo, tags=["Cloud"])
