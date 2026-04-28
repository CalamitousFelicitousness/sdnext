from __future__ import annotations
import io
import json
import base64
from typing import Iterator, Optional
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from modules.logger import log
from modules.cloud import (
    TextRequest,
    VisionRequest,
    ImageRequest,
    VideoRequest,
    Job,
    predict_text as cloud_predict_text,
    predict_vision as cloud_predict_vision,
    stream_text as cloud_stream_text,
    submit_job,
    get_job,
    list_jobs,
    cancel_job,
    list_providers,
    list_models,
    get_handler,
)
from modules.api.cloud_models import (
    CloudProviderInfo,
    CloudModelInfo,
    CloudTextRequest,
    CloudVisionRequest,
    CloudTextResponse,
    CloudImageRequest,
    CloudVideoRequest,
    CloudImageResultInline,
    CloudVideoResultInline,
    CloudJob,
)


CAPABILITY_ORDER = ('text', 'vision', 'image', 'video')


def make_provider_info(capability: str, provider_id: str) -> CloudProviderInfo:
    entries = {cap: get_handler(cap, provider_id) or {} for cap in CAPABILITY_ORDER}
    capabilities = [cap for cap in CAPABILITY_ORDER if entries[cap]]
    label = next((entries[cap].get('label') for cap in CAPABILITY_ORDER if entries[cap].get('label')), provider_id)
    enabled_fn = next((entries[cap].get('enabled') for cap in CAPABILITY_ORDER if entries[cap].get('enabled')), lambda: False)
    try:
        enabled = bool(enabled_fn())
    except Exception:
        enabled = False
    primary = capability if capability in CAPABILITY_ORDER and entries[capability] else (capabilities[0] if capabilities else 'text')
    models = list_models(primary, provider_id)
    return CloudProviderInfo(
        id=provider_id,
        label=label,
        capabilities=capabilities,
        enabled=enabled,
        models=list(models),
    )


def get_providers() -> list[CloudProviderInfo]:
    """List all configured cloud providers and their declared capabilities/models.

    Returns one entry per provider id known to the registry. The ``enabled``
    field reflects whether the provider has API credentials configured. Models
    listed are the framework's default set; provider-specific dropdowns may
    accept additional values.
    """
    seen = set()
    out: list[CloudProviderInfo] = []
    for cap in CAPABILITY_ORDER:
        for pid in list_providers(cap, only_enabled=False):
            if pid in seen:
                continue
            seen.add(pid)
            out.append(make_provider_info(cap, pid))
    return out


def get_provider_models(provider_id: str) -> list[CloudModelInfo]:
    """List default model identifiers for a single provider.

    Combines text and vision capabilities — a model is reported as vision-capable
    if the provider exposes a vision handler that lists it.
    """
    text_entry = get_handler('text', provider_id)
    vision_entry = get_handler('vision', provider_id)
    if text_entry is None and vision_entry is None:
        raise HTTPException(status_code=404, detail=f'Unknown provider: {provider_id}')
    text_models = set(list_models('text', provider_id))
    vision_models = set(list_models('vision', provider_id))
    streamer_supported = bool(text_entry and text_entry.get('stream'))
    all_ids = sorted(text_models | vision_models)
    return [
        CloudModelInfo(
            id=mid,
            supports_vision=mid in vision_models,
            supports_streaming=streamer_supported and mid in text_models,
        )
        for mid in all_ids
    ]


def decode_image(payload: str) -> Image.Image:
    if not payload:
        raise HTTPException(status_code=400, detail='image field is required')
    if payload.startswith('data:'):
        try:
            _, b64 = payload.split(',', 1)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f'malformed data URL: {e}') from e
    else:
        b64 = payload
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'invalid base64 image: {e}') from e
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'cannot open image: {e}') from e


def to_text_request(req: CloudTextRequest) -> TextRequest:
    from modules.caption.models_def import resolve_provider, strip_provider_prefix  # pylint: disable=import-outside-toplevel
    provider_id = resolve_provider(req.model) or req.provider
    clean_model = strip_provider_prefix(req.model, provider_id) if provider_id else req.model
    return TextRequest(
        model=clean_model,
        prompt=req.prompt,
        system=req.system,
        prefill=req.prefill,
        thinking=req.thinking,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        top_k=req.top_k,
        stream=req.stream,
        extra=req.extra or {},
    )


def to_vision_request(req: CloudVisionRequest) -> VisionRequest:
    base = to_text_request(req)
    return VisionRequest(
        model=base.model,
        prompt=base.prompt,
        system=base.system,
        prefill=base.prefill,
        thinking=base.thinking,
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        top_p=base.top_p,
        top_k=base.top_k,
        stream=base.stream,
        extra=dict(base.extra),
        image=decode_image(req.image),
    )


def post_text(req: CloudTextRequest) -> CloudTextResponse:
    """Run a text-only completion against the named provider.

    The framework strips matching provider prefixes from the model field so
    callers may pass either the bare provider model id or a namespaced form
    (e.g. ``openrouter/anthropic/claude-3.5-sonnet``).
    """
    if get_handler('text', req.provider) is None:
        raise HTTPException(status_code=404, detail=f'Unknown text provider: {req.provider}')
    treq = to_text_request(req)
    resp = cloud_predict_text(req.provider, treq)
    return CloudTextResponse(
        text=resp.text or '',
        finish_reason=resp.finish_reason,
        usage=resp.usage,
        model=resp.model,
        error=resp.error,
    )


def post_vision(req: CloudVisionRequest) -> CloudTextResponse:
    """Run a vision+text completion against the named provider.

    The ``image`` field accepts either a bare base64 string or a full data URL.
    Images are forwarded to providers in their native shape (Anthropic uses a
    base64 source block; OpenAI-compat providers use a data URL).
    """
    if get_handler('vision', req.provider) is None:
        raise HTTPException(status_code=404, detail=f'Unknown vision provider: {req.provider}')
    vreq = to_vision_request(req)
    resp = cloud_predict_vision(req.provider, vreq)
    return CloudTextResponse(
        text=resp.text or '',
        finish_reason=resp.finish_reason,
        usage=resp.usage,
        model=resp.model,
        error=resp.error,
    )


def post_text_stream(req: CloudTextRequest) -> StreamingResponse:
    """Stream a text completion as SSE events.

    Each chunk is encoded as ``data: {"delta":"...","done":false}\\n\\n``. The
    final event is ``data: {"delta":"","done":true}\\n\\n``. On error mid-stream
    a final event ``{"delta":"","done":true,"error":"..."}`` is emitted before
    the connection closes.
    """
    if get_handler('text', req.provider) is None:
        raise HTTPException(status_code=404, detail=f'Unknown text provider: {req.provider}')
    treq = to_text_request(req)
    treq.stream = True

    def event_stream() -> Iterator[bytes]:
        try:
            for chunk in cloud_stream_text(req.provider, treq):
                payload = {'delta': chunk, 'done': False}
                yield f'data: {json.dumps(payload)}\n\n'.encode('utf-8')
            yield b'data: {"delta": "", "done": true}\n\n'
        except Exception as e:
            log.warning(f'Cloud stream error: provider={req.provider} error={e}')
            err_payload = {'delta': '', 'done': True, 'error': str(e)}
            yield f'data: {json.dumps(err_payload)}\n\n'.encode('utf-8')

    return StreamingResponse(event_stream(), media_type='text/event-stream')


def to_image_request(req: CloudImageRequest) -> ImageRequest:
    from modules.caption.models_def import resolve_provider, strip_provider_prefix  # pylint: disable=import-outside-toplevel
    provider_id = resolve_provider(req.model) or req.provider
    clean_model = strip_provider_prefix(req.model, provider_id) if provider_id else req.model
    return ImageRequest(
        model=clean_model,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        num_images=req.num_images,
        init_image=decode_image(req.init_image) if req.init_image else None,
        mask=decode_image(req.mask) if req.mask else None,
        strength=req.strength,
        extra=req.extra or {},
    )


def to_video_request(req: CloudVideoRequest) -> VideoRequest:
    from modules.caption.models_def import resolve_provider, strip_provider_prefix  # pylint: disable=import-outside-toplevel
    provider_id = resolve_provider(req.model) or req.provider
    clean_model = strip_provider_prefix(req.model, provider_id) if provider_id else req.model
    return VideoRequest(
        model=clean_model,
        prompt=req.prompt,
        duration=req.duration,
        width=req.width,
        height=req.height,
        fps=req.fps,
        seed=req.seed,
        image=decode_image(req.image) if req.image else None,
        num_frames=req.num_frames,
        extra=req.extra or {},
    )


def encode_image_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("ascii")}'


def to_cloud_job(job: Job) -> CloudJob:
    result = None
    if job.result is not None:
        if job.capability == 'image':
            try:
                imgs = [encode_image_data_url(im) for im in (getattr(job.result, 'images', None) or []) if im is not None]
            except Exception as e:
                log.debug(f'Cloud: image result encode failed for {job.id}: {e}')
                imgs = []
            result = CloudImageResultInline(
                images=imgs,
                finish_reason=getattr(job.result, 'finish_reason', None),
                model=getattr(job.result, 'model', None),
            ).model_dump()
        elif job.capability == 'video':
            video_b64 = None
            video_bytes = getattr(job.result, 'video_bytes', None)
            if video_bytes:
                try:
                    video_b64 = base64.b64encode(video_bytes).decode('ascii')
                except Exception as e:
                    log.debug(f'Cloud: video result encode failed for {job.id}: {e}')
            result = CloudVideoResultInline(
                video_b64=video_b64,
                video_path=getattr(job.result, 'video_path', None),
                duration=getattr(job.result, 'duration', None),
                model=getattr(job.result, 'model', None),
            ).model_dump()
    return CloudJob(
        id=job.id,
        provider_id=job.provider_id,
        capability=job.capability,
        status=job.status,
        progress=job.progress,
        message=job.message or '',
        error=job.error,
        started_at=job.started_at,
        updated_at=job.updated_at,
        result=result,
    )


def post_image(req: CloudImageRequest) -> CloudJob:
    """Submit an image generation job and return its initial state.

    Always returns a job — sync providers (e.g. NanoBanana) complete near-instantly,
    async providers return ``status='submitted'``. Poll
    ``GET /sdapi/v1/cloud/jobs/{id}`` until ``status`` is terminal.
    """
    if get_handler('image', req.provider) is None:
        raise HTTPException(status_code=404, detail=f'Unknown image provider: {req.provider}')
    ireq = to_image_request(req)
    job = submit_job('image', req.provider, ireq)
    return to_cloud_job(job)


def post_video(req: CloudVideoRequest) -> CloudJob:
    """Submit a video generation job and return its initial state.

    Long-running providers (Veo) drive a poll loop in a worker thread. Poll
    ``GET /sdapi/v1/cloud/jobs/{id}`` until ``status`` is terminal.
    """
    if get_handler('video', req.provider) is None:
        raise HTTPException(status_code=404, detail=f'Unknown video provider: {req.provider}')
    vreq = to_video_request(req)
    job = submit_job('video', req.provider, vreq)
    return to_cloud_job(job)


def get_jobs(capability: Optional[str] = None, status: Optional[str] = None) -> list[CloudJob]:
    """List all jobs (both running and completed history).

    Optional filters ``?capability=image|video`` and ``?status=running|succeeded|...``.
    Sorted by ``started_at`` descending.
    """
    return [to_cloud_job(j) for j in list_jobs(capability=capability, status=status)]


def get_job_state(job_id: str) -> CloudJob:
    """Fetch a single job by id."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Unknown job id: {job_id}')
    return to_cloud_job(job)


def cancel_job_route(job_id: str) -> CloudJob:
    """Request cancellation of an in-flight job.

    The job is marked for cancellation between polls; for async providers the
    framework dispatches the provider's ``cancel`` callable fire-and-forget.
    Already-terminal jobs return their existing state without modification.
    """
    ok = cancel_job(job_id)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Unknown job id: {job_id}')
    if not ok:
        log.debug(f'Cloud: cancel_job had no effect (already terminal?) for {job_id}')
    return to_cloud_job(job)


def register_api(api):
    api.add_api_route("/sdapi/v1/cloud/providers", get_providers, methods=["GET"], response_model=list[CloudProviderInfo], tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/providers/{provider_id}/models", get_provider_models, methods=["GET"], response_model=list[CloudModelInfo], tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/text", post_text, methods=["POST"], response_model=CloudTextResponse, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/vision", post_vision, methods=["POST"], response_model=CloudTextResponse, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/text/stream", post_text_stream, methods=["POST"], tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/image", post_image, methods=["POST"], response_model=CloudJob, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/video", post_video, methods=["POST"], response_model=CloudJob, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/jobs", get_jobs, methods=["GET"], response_model=list[CloudJob], tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/jobs/{job_id}", get_job_state, methods=["GET"], response_model=CloudJob, tags=["Cloud"])
    api.add_api_route("/sdapi/v1/cloud/jobs/{job_id}/cancel", cancel_job_route, methods=["POST"], response_model=CloudJob, tags=["Cloud"])
