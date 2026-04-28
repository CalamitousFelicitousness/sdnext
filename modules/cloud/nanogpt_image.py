"""NanoGPT image generation cloud provider.

Registers `nanogpt` against `IMAGE_PROVIDERS` (mode='async'). One uniform
async registration covers two NanoGPT-side paths:

  - **Sync inline**: most models (`hidream`, `flux-kontext`, `gpt-image-1`,
    `gpt-4o-image`, `flux-lora/inpainting`, ...) return inline images on the
    first POST. ``submit`` finishes the job in one round-trip; the framework
    poll loop never executes because the job is already terminal.

  - **Async Midjourney**: ``midjourney-*`` models return a ``task_id`` from
    the same submit endpoint. ``submit`` stashes the task id in
    ``job.extra['task_id']`` and the framework drives ``poll`` against
    ``POST /api/check-midjourney-status`` until the task reports ``SUCCESS``,
    ``FAILED``, or stays in a transient state.

NanoGPT does not expose a server-side cancel for image jobs; ``cancel`` is
unregistered (the framework's local-cancel marks the job cancelled and stops
polling, while the upstream task continues to bill — same contract as Veo).
"""
from __future__ import annotations
import base64
import io
from typing import TYPE_CHECKING
from PIL import Image
from modules.logger import log
from modules.cloud import client, client_async
from modules.cloud.nanogpt import (
    V1_BASE,
    API_BASE,
    IMAGE_FALLBACK_MODELS,
    auth_headers,
    is_enabled,
    list_models_cached,
)
from modules.cloud.types import ImageRequest, ImageResponse, Job
from modules.cloud.registry import register_image


if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


PROVIDER_ID = 'nanogpt'
LABEL = 'NanoGPT'

DEFAULT_POLL_INTERVAL = 5.0

SIZE_BUCKETS = ('256x256', '512x512', '1024x1024')


def list_default_models() -> list[str]:
    return list_models_cached('image', IMAGE_FALLBACK_MODELS)


def closest_size(width: int, height: int) -> str:
    target = max(width or 0, height or 0) or 1024
    return min(SIZE_BUCKETS, key=lambda s: abs(int(s.split('x')[0]) - target))


def is_midjourney(model: str) -> bool:
    return (model or '').lower().startswith('midjourney')


def build_body(req: ImageRequest) -> dict:
    body: dict = {
        'model': req.model,
        'prompt': req.prompt,
        'n': max(1, int(req.num_images or 1)),
        'size': closest_size(req.width, req.height),
        'response_format': 'b64_json',
    }
    if req.seed is not None:
        body['seed'] = req.seed
    if req.guidance_scale is not None:
        body['guidance_scale'] = float(req.guidance_scale)
    if req.steps and req.steps != 30:
        body['num_inference_steps'] = int(req.steps)
    if req.strength is not None:
        body['strength'] = float(req.strength)
    if req.init_image is not None:
        body['imageDataUrl'] = client.image_to_data_url(req.init_image, max_dim=2048)
    if req.mask is not None:
        body['maskDataUrl'] = client.image_to_data_url(req.mask, max_dim=2048)
    extra = req.extra or {}
    body.update(extra)
    return body


def decode_b64(b64: str) -> 'PILImage':
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


async def fetch_url(async_client, url: str) -> 'PILImage':
    raw = await client_async.download_async(async_client, url, headers=auth_headers())
    return Image.open(io.BytesIO(raw))


async def decode_entry(entry: dict, async_client) -> 'PILImage':
    if entry.get('b64_json'):
        return decode_b64(entry['b64_json'])
    if entry.get('url'):
        return await fetch_url(async_client, entry['url'])
    raise RuntimeError('image entry has neither b64_json nor url')


def extract_task_id(resp: dict) -> str:
    """Defensive parse — docs are ambiguous about top-level vs data[0] placement."""
    if resp.get('task_id'):
        return str(resp['task_id'])
    data = resp.get('data')
    if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get('task_id'):
        return str(data[0]['task_id'])
    if isinstance(data, dict) and data.get('task_id'):
        return str(data['task_id'])
    return ''


async def submit(job: Job) -> None:
    req: ImageRequest = job.request
    body = build_body(req)
    log.debug(f'Cloud: provider={PROVIDER_ID} cap=image model="{req.model}" submit n={body.get("n")} size={body.get("size")}')
    async with client_async.make_client(timeout=120.0) as cx:
        try:
            resp = await client_async.post_json_async(cx, f'{V1_BASE}/images/generations', auth_headers(), body)
        except Exception as e:
            job.status = 'failed'
            job.error = f'submit failed: {e}'
            return

        if is_midjourney(req.model):
            task_id = extract_task_id(resp)
            if task_id:
                job.extra['task_id'] = task_id
                job.status = 'submitted'
                job.progress = max(job.progress, 0.1)
                job.message = 'midjourney queued'
                return
            log.warning(f'Cloud: provider={PROVIDER_ID} model="{req.model}" no task_id; falling back to inline parse')

        entries = resp.get('data') or []
        if not entries:
            job.status = 'failed'
            job.error = 'no data in response'
            return
        images: list = []
        for entry in entries:
            try:
                images.append(await decode_entry(entry, cx))
            except Exception as e:
                log.warning(f'Cloud: provider={PROVIDER_ID} entry decode failed: {e}')
        if not images:
            job.status = 'failed'
            job.error = 'all image entries failed to decode'
            return
        job.result = ImageResponse(images=images, model=req.model)
        job.status = 'succeeded'
        job.progress = 1.0
        job.message = 'completed'


async def poll(job: Job) -> None:
    """Drives the Midjourney status path. No-op-effect if submit already terminal."""
    task_id = job.extra.get('task_id')
    if not task_id:
        # framework only enters poll while status is non-terminal; if we land here
        # without a task_id, mark failed loudly so it surfaces in REST + WS.
        job.status = 'failed'
        job.error = 'poll called without task_id'
        return
    async with client_async.make_client(timeout=60.0) as cx:
        try:
            resp = await client_async.post_json_async(cx, f'{API_BASE}/check-midjourney-status',
                                                     auth_headers(), {'task_id': task_id})
        except Exception as e:
            job.status = 'failed'
            job.error = f'poll failed: {e}'
            return
        status = (resp.get('status') or '').upper()
        prog = resp.get('progress')
        if isinstance(prog, (int, float)):
            normalized = float(prog) / 100.0 if prog > 1 else float(prog)
            job.progress = max(job.progress, 0.0, min(1.0, normalized))
        if status == 'SUCCESS':
            url = resp.get('imageUrl')
            if not url:
                job.status = 'failed'
                job.error = 'success without imageUrl'
                return
            try:
                img = await fetch_url(cx, url)
            except Exception as e:
                job.status = 'failed'
                job.error = f'download failed: {e}'
                return
            job.result = ImageResponse(images=[img], model=job.request.model)
            job.status = 'succeeded'
            job.progress = 1.0
            job.message = 'completed'
        elif status == 'FAILED':
            job.status = 'failed'
            job.error = resp.get('failReason') or 'midjourney failed'
        elif status in ('RUNNING', 'IN_PROGRESS'):
            job.status = 'running'
            job.message = 'generating'
        else:  # PENDING, NOT_START, SUBMITTED, UNKNOWN, ''
            job.status = 'submitted'
            job.message = status.lower() or 'pending'


register_image(
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
