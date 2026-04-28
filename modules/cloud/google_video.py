"""Google Veo (video-generation) cloud provider.

Registers the ``google`` provider against ``VIDEO_PROVIDERS`` (mode='async'),
and exposes a sync ``GoogleVeoVideoPipeline`` shim that preserves the legacy
``processing_diffusers.py`` invocation — the call returns
``{'bytes': <mp4 bytes>, 'images': []}`` or ``None`` to keep the video
downstream pipeline unchanged.

The ``submit/poll/cancel`` triple wraps the synchronous google-genai SDK in
``asyncio.to_thread`` so the runner's event loop can observe interrupts and
the watchdog between polls without blocking on network I/O.
"""
from __future__ import annotations
import asyncio
import io
import time
from typing import TYPE_CHECKING
from modules.logger import log
from modules.cloud.types import VideoRequest, VideoResponse, Job, TERMINAL_JOB_STATUSES
from modules.cloud.registry import register_video
from modules.cloud.google import (
    PROVIDER_ID,
    LABEL,
    get_client,
    is_enabled,
    normalize_model,
)


if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


DEFAULT_MODELS = (
    'veo-3.1-generate-preview',
)


VIDEO_RESOLUTION_BUCKETS = {
    '720p': 1280 * 720,
    '1080p': 1920 * 1080,
}
ASPECT_RATIO_BUCKETS = {
    '1:1':  1 / 1,
    '2:3':  2 / 3,
    '3:2':  3 / 2,
    '4:3':  4 / 3,
    '3:4':  3 / 4,
    '4:5':  4 / 5,
    '5:4':  5 / 4,
    '16:9': 16 / 9,
    '9:16': 9 / 16,
    '21:9': 21 / 9,
    '9:21': 9 / 21,
}

# Veo accepts integer duration_seconds in [4, 8]; clamp here to avoid 400s.
VEO_MIN_DURATION = 4
VEO_MAX_DURATION = 8

DEFAULT_POLL_INTERVAL = 10.0


def list_default_models() -> list[str]:
    return list(DEFAULT_MODELS)


def get_size_buckets(width: int, height: int) -> tuple[str, str]:
    aspect_ratio = width / height
    pixel_count = width * height
    closest_resolution = min(VIDEO_RESOLUTION_BUCKETS.items(), key=lambda x: abs(x[1] - pixel_count))[0]
    closest_aspect = min(ASPECT_RATIO_BUCKETS.items(), key=lambda x: abs(x[1] - aspect_ratio))[0]
    return closest_resolution, closest_aspect


def derive_duration(req: VideoRequest) -> int:
    if req.duration is not None:
        seconds = int(round(req.duration))
    elif req.num_frames is not None and req.fps:
        seconds = int(round(req.num_frames / max(req.fps, 1)))
    elif req.num_frames is not None:
        seconds = req.num_frames // 24
    else:
        seconds = VEO_MIN_DURATION
    return max(VEO_MIN_DURATION, min(VEO_MAX_DURATION, seconds))


def build_config(req: VideoRequest):
    from google.genai import types  # pylint: disable=import-outside-toplevel,no-name-in-module
    resolution, aspect_ratio = get_size_buckets(req.width, req.height)
    duration_seconds = derive_duration(req)
    return types.GenerateVideosConfig(
        duration_seconds=duration_seconds,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )


def kick_operation(client, model: str, prompt: str, config, image: 'PILImage' = None):
    from google.genai import types  # pylint: disable=import-outside-toplevel,no-name-in-module
    if image is not None:
        save_img = image if image.mode == 'RGB' else image.convert('RGB')
        buf = io.BytesIO()
        save_img.save(buf, format='JPEG')
        return client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=config,
            image=types.Image(image_bytes=buf.getvalue(), mime_type='image/jpeg'),
        )
    return client.models.generate_videos(model=model, prompt=prompt, config=config)


async def submit(job: Job) -> None:
    req: VideoRequest = job.request
    client = get_client()
    if client is None:
        job.status = 'failed'
        job.error = f'No credentials configured for {LABEL}'
        return
    model = normalize_model(req.model)
    job.extra['model'] = model
    config = build_config(req)
    job.message = 'submitting'
    log.debug(f'Cloud: provider={PROVIDER_ID} cap=video model="{model}" submit')
    operation = await asyncio.to_thread(kick_operation, client, model, req.prompt, config, req.image)
    job.extra['operation'] = operation
    job.status = 'submitted'
    job.message = 'submitted'
    job.progress = 0.05


async def poll(job: Job) -> None:
    operation = job.extra.get('operation')
    if operation is None:
        job.status = 'failed'
        job.error = 'missing operation handle'
        return
    client = get_client()
    if client is None:
        job.status = 'failed'
        job.error = f'No credentials configured for {LABEL}'
        return
    model = job.extra.get('model') or normalize_model(job.request.model)
    operation = await asyncio.to_thread(client.operations.get, operation)
    job.extra['operation'] = operation
    if not getattr(operation, 'done', False):
        # SDK does not expose a percentage; report a steady "running" half-progress so the UI
        # has something to render rather than 0% throughout the multi-minute job.
        job.status = 'running'
        job.progress = max(job.progress, 0.5)
        job.message = 'generating'
        return
    response_obj = getattr(operation, 'response', None)
    generated = getattr(response_obj, 'generated_videos', None) if response_obj is not None else None
    if not generated:
        job.status = 'failed'
        job.error = 'no generated_videos in response'
        return
    try:
        first = generated[0]
        await asyncio.to_thread(client.files.download, file=first.video)
        video_bytes = getattr(first.video, 'video_bytes', None)
        if not video_bytes:
            job.status = 'failed'
            job.error = 'video bytes missing after download'
            return
        job.result = VideoResponse(
            video_bytes=video_bytes,
            duration=getattr(job.request, 'duration', None),
            model=model,
        )
        job.status = 'succeeded'
        job.progress = 1.0
        job.message = 'completed'
    except Exception as e:
        log.error(f'Cloud: provider={PROVIDER_ID} cap=video download failed: {e}')
        job.status = 'failed'
        job.error = f'download failed: {e}'


async def cancel(job: Job) -> None:
    operation = job.extra.get('operation')
    if operation is None:
        return
    client = get_client()
    if client is None:
        return

    def do_cancel():
        try:
            client.operations.cancel(operation)
        except Exception as e:
            log.warning(f'Cloud: provider={PROVIDER_ID} cap=video cancel failed: {e}')

    await asyncio.to_thread(do_cancel)


class GoogleVeoVideoPipeline:
    """Sync wrapper preserving the legacy ``video_load.load_custom`` contract.

    Submits a Job, polls JOBS until terminal, returns ``{'bytes': ..., 'images': []}``
    on success or ``None`` on failure / cancellation / timeout.
    """

    def __init__(self, model_name: str):
        self.model = model_name
        log.debug(f'Load model: type=GoogleVeo model="{model_name}"')

    def __call__(self, prompt, width: int, height: int, image=None, num_frames: int = 96):
        from modules.cloud import jobs  # pylint: disable=import-outside-toplevel
        from modules import shared  # pylint: disable=import-outside-toplevel
        text = prompt[0] if isinstance(prompt, (list, tuple)) and prompt else (prompt or '')
        req = VideoRequest(model=self.model, prompt=text, width=width, height=height, image=image, num_frames=num_frames)
        job = jobs.submit_job('video', PROVIDER_ID, req)
        watchdog = float(getattr(shared.opts, 'cloud_job_max_duration', 600.0)) + 30.0
        deadline = time.time() + watchdog
        while time.time() < deadline:
            current = jobs.get_job(job.id)
            if current is None:
                return None
            if current.status in TERMINAL_JOB_STATUSES:
                if current.status == 'succeeded' and current.result is not None and current.result.video_bytes:
                    return {'bytes': current.result.video_bytes, 'images': []}
                if current.error:
                    log.warning(f'Cloud: provider={PROVIDER_ID} job={job.id} error={current.error}')
                return None
            time.sleep(2)
        log.warning(f'Cloud: provider={PROVIDER_ID} job={job.id} watchdog timed out — cancelling')
        jobs.cancel_job(job.id)
        return None


def build_pipeline(model_name: str) -> GoogleVeoVideoPipeline:
    return GoogleVeoVideoPipeline(model_name)


register_video(
    PROVIDER_ID,
    mode='async',
    submit=submit,
    poll=poll,
    cancel=cancel,
    poll_interval=DEFAULT_POLL_INTERVAL,
    label=LABEL,
    models=list_default_models,
    enabled=is_enabled,
)
