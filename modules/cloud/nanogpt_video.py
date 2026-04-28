"""NanoGPT video generation cloud provider.

Registers `nanogpt` against `VIDEO_PROVIDERS` (mode='async') for the full
NanoGPT video catalog (~28 models across Veo, Kling, MiniMax, Hunyuan, Wan,
Wavespeed, Seedance, Pixverse, Lightricks, Vidu, Runway and utility
upscalers).

Three-step async contract (verified against docs.nano-gpt.com):

  1. ``submit``: ``POST /api/generate-video`` returns ``{ runId, id, status:
     'pending', model, cost }``. Stash ``runId`` in ``job.extra``.
  2. ``poll``: ``GET /api/video/status?requestId={runId}`` returns
     ``{ data: { status, output?: { video: { url } }, ...errors } }``.
     Translate uppercase poll-side enum (`IN_QUEUE`/`IN_PROGRESS`/
     `COMPLETED`/`FAILED`/`CANCELED`) into framework `JobStatus`.
  3. On `COMPLETED`, download the mp4 via ``download_async`` and store as
     ``VideoResponse.video_bytes``.

NanoGPT does not document a cancel endpoint; ``cancel`` is unregistered.
The framework's local cancel marks the job cancelled and stops polling
(upstream task continues, same contract as Veo and the image provider).

Optional fields are gathered from ``req.extra`` so per-model knobs (Veo
``generateAudio``, Kling ``cfg_scale``, Wan ``num_frames``) flow through
without a per-model schema.
"""
from __future__ import annotations
from modules.logger import log
from modules.cloud import client, client_async
from modules.cloud.nanogpt import (
    API_BASE,
    VIDEO_FALLBACK_MODELS,
    auth_headers,
    is_enabled,
    list_models_cached,
)
from modules.cloud.types import VideoRequest, VideoResponse, Job
from modules.cloud.registry import register_video


PROVIDER_ID = 'nanogpt'
LABEL = 'NanoGPT'

DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_DOWNLOAD_TIMEOUT = 300.0


ASPECT_RATIOS = {
    '16:9': 16 / 9,
    '9:16': 9 / 16,
    '1:1':  1.0,
    '4:3':  4 / 3,
    '3:4':  3 / 4,
}

DEFAULT_DURATION_SECONDS = 5


def list_default_models() -> list[str]:
    return list_models_cached('video', VIDEO_FALLBACK_MODELS)


def derive_duration(req: VideoRequest) -> str:
    if req.duration is not None:
        seconds = int(round(req.duration))
    elif req.num_frames and req.fps:
        seconds = int(round(req.num_frames / max(req.fps, 1)))
    elif req.num_frames:
        seconds = req.num_frames // 24
    else:
        seconds = DEFAULT_DURATION_SECONDS
    return str(max(1, seconds))


def closest_aspect_ratio(width: int, height: int) -> str:
    target = (width or 1) / max(height or 1, 1)
    return min(ASPECT_RATIOS, key=lambda k: abs(ASPECT_RATIOS[k] - target))


def closest_resolution(width: int, height: int) -> str:
    pixels = (width or 0) * (height or 0)
    if pixels >= 1280 * 720 * 1.5:
        return '1080p'
    if pixels >= 640 * 480:
        return '720p'
    return '480p'


def build_body(req: VideoRequest) -> dict:
    body: dict = {
        'model': req.model,
        'prompt': req.prompt,
        'duration': derive_duration(req),
        'aspect_ratio': closest_aspect_ratio(req.width, req.height),
        'resolution': closest_resolution(req.width, req.height),
    }
    if req.seed is not None:
        body['seed'] = req.seed
    if req.image is not None:
        body['imageDataUrl'] = client.image_to_data_url(req.image, max_dim=2048)
    extra = req.extra or {}
    if 'negative_prompt' in extra:
        body['negative_prompt'] = extra['negative_prompt']
    if 'generateAudio' in extra:
        body['generateAudio'] = bool(extra['generateAudio'])
    if 'pro_mode' in extra or 'pro' in extra:
        body['pro_mode'] = bool(extra.get('pro_mode', extra.get('pro')))
    if 'cfg_scale' in extra:
        try:
            body['cfg_scale'] = float(extra['cfg_scale'])
        except (TypeError, ValueError):
            pass
    if 'num_frames' in extra:
        try:
            body['num_frames'] = int(extra['num_frames'])
        except (TypeError, ValueError):
            pass
    if 'frames_per_second' in extra:
        try:
            body['frames_per_second'] = int(extra['frames_per_second'])
        except (TypeError, ValueError):
            pass
    if 'num_inference_steps' in extra:
        try:
            body['num_inference_steps'] = int(extra['num_inference_steps'])
        except (TypeError, ValueError):
            pass
    if 'camera_fixed' in extra:
        body['camera_fixed'] = bool(extra['camera_fixed'])
    return body


async def submit(job: Job) -> None:
    req: VideoRequest = job.request
    body = build_body(req)
    log.debug(f'Cloud: provider={PROVIDER_ID} cap=video model="{req.model}" submit duration={body.get("duration")}s ar={body.get("aspect_ratio")} res={body.get("resolution")}')
    async with client_async.make_client(timeout=60.0) as cx:
        try:
            resp = await client_async.post_json_async(cx, f'{API_BASE}/generate-video', auth_headers(), body)
        except Exception as e:
            job.status = 'failed'
            job.error = f'submit failed: {e}'
            return
    run_id = resp.get('runId') or resp.get('id')
    if not run_id:
        job.status = 'failed'
        job.error = 'no runId in response'
        return
    job.extra['run_id'] = run_id
    job.extra['model'] = req.model
    job.status = 'submitted'
    job.progress = max(job.progress, 0.05)
    job.message = (resp.get('status') or 'pending').lower()


async def poll(job: Job) -> None:
    run_id = job.extra.get('run_id')
    if not run_id:
        job.status = 'failed'
        job.error = 'missing run_id'
        return
    async with client_async.make_client(timeout=60.0) as cx:
        try:
            resp = await client_async.get_json_async(
                cx,
                f'{API_BASE}/video/status',
                auth_headers(),
                params={'requestId': run_id},
            )
        except Exception as e:
            job.status = 'failed'
            job.error = f'poll failed: {e}'
            return
        data = resp.get('data') or {}
        status = (data.get('status') or '').upper()
        if status == 'COMPLETED':
            output = data.get('output') or {}
            video = output.get('video') or {}
            url = video.get('url')
            if not url:
                job.status = 'failed'
                job.error = 'completed without video url'
                return
            try:
                video_bytes = await client_async.download_async(
                    cx, url, headers=auth_headers(), timeout=DEFAULT_DOWNLOAD_TIMEOUT,
                )
            except Exception as e:
                job.status = 'failed'
                job.error = f'download failed: {e}'
                return
            req: VideoRequest = job.request
            job.result = VideoResponse(
                video_bytes=video_bytes,
                duration=req.duration,
                model=req.model,
            )
            job.status = 'succeeded'
            job.progress = 1.0
            job.message = 'completed'
        elif status == 'FAILED':
            job.status = 'failed'
            job.error = data.get('userFriendlyError') or data.get('error') or 'video generation failed'
        elif status == 'CANCELED':
            job.status = 'cancelled'
            job.message = 'cancelled by provider'
        elif status == 'IN_PROGRESS':
            job.status = 'running'
            job.progress = max(job.progress, 0.5)
            job.message = 'generating'
        elif status == 'IN_QUEUE':
            job.status = 'submitted'
            job.progress = max(job.progress, 0.1)
            job.message = 'queued'
        else:
            # Unknown status — keep job alive in 'running' so we re-poll rather than terminate spuriously
            job.status = 'running'
            job.message = (status or 'unknown').lower()


register_video(
    PROVIDER_ID,
    mode='async',
    submit=submit,
    poll=poll,
    cancel=None,
    poll_interval=DEFAULT_POLL_INTERVAL,
    label=LABEL,
    models=list_default_models,
    enabled=is_enabled,
)
