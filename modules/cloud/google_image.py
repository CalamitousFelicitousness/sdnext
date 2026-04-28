"""Google NanoBanana (Gemini image-generation) cloud provider.

Registers the ``google`` provider against ``IMAGE_PROVIDERS`` and exposes a
sync ``GoogleNanoBananaPipeline`` shim that preserves the legacy
``processing_diffusers.py:187`` contract — ``output = shared.sd_model(**args)``.

Three sites dispatch by the literal class name ``GoogleNanoBananaPipeline``
and must keep working unchanged:

  - ``modules/sd_models.py:54`` (class registry)
  - ``modules/modeldata.py:138-144`` (model_type detection)
  - ``modules/processing_args.py:158`` (init_image plumbing)
"""
from __future__ import annotations
import asyncio
import io
import time
from typing import TYPE_CHECKING
from PIL import Image
from modules.logger import log
from modules.cloud.types import ImageRequest, ImageResponse, TERMINAL_JOB_STATUSES
from modules.cloud.registry import register_image
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
    'gemini-3-pro-image-preview',
    'gemini-2.5-flash-image-preview',
)


IMAGE_SIZE_BUCKETS = {
    '1K': 1024 * 1024,
    '2K': 2048 * 1024,
    '4K': 4096 * 1024,
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


# Compile-time invariant: literal-name dispatch in sd_models.py / modeldata.py / processing_args.py
# breaks silently if this name drifts. Catch it loud at import.
SENTINEL_CLASS_NAME = 'GoogleNanoBananaPipeline'


def list_default_models() -> list[str]:
    return list(DEFAULT_MODELS)


def get_size_buckets(width: int, height: int) -> tuple[str, str]:
    aspect_ratio = width / height
    pixel_count = width * height
    closest_size = min(IMAGE_SIZE_BUCKETS.items(), key=lambda x: abs(x[1] - pixel_count))[0]
    closest_aspect = min(ASPECT_RATIO_BUCKETS.items(), key=lambda x: abs(x[1] - aspect_ratio))[0]
    return closest_size, closest_aspect


def build_image_config(model: str, image_size: str, aspect_ratio: str):
    from google.genai import types  # pylint: disable=import-outside-toplevel,no-name-in-module
    if 'gemini-3' in model:
        return types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size)
    return types.ImageConfig(aspect_ratio=aspect_ratio)


def call_model(client, model: str, prompt: str, image_config, image: 'PILImage' = None):
    from google.genai import types  # pylint: disable=import-outside-toplevel,no-name-in-module
    config = types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=image_config)
    if image is not None:
        save_img = image if image.mode == 'RGB' else image.convert('RGB')
        buf = io.BytesIO()
        save_img.save(buf, format='JPEG')
        contents = [types.Part.from_bytes(data=buf.getvalue(), mime_type='image/jpeg'), prompt]
    else:
        contents = prompt
    return client.models.generate_content(model=model, contents=contents, config=config)


def extract_images(response) -> list['PILImage']:
    out: list['PILImage'] = []
    if getattr(response, 'prompt_feedback', None) is not None:
        log.warning(f'Cloud: provider={PROVIDER_ID} feedback={response.prompt_feedback}')
    candidates = getattr(response, 'candidates', None) or []
    for candidate in candidates:
        for part in getattr(getattr(candidate, 'content', None), 'parts', None) or []:
            inline = getattr(part, 'inline_data', None)
            if inline is not None and getattr(inline, 'data', None):
                try:
                    out.append(Image.open(io.BytesIO(inline.data)))
                except Exception as e:
                    log.warning(f'Cloud: provider={PROVIDER_ID} image decode failed: {e}')
    return out


async def predict_image(req: ImageRequest) -> ImageResponse:
    """Async sync-mode predict: dispatches to google-genai SDK on a worker thread."""

    def run() -> ImageResponse:
        client = get_client()
        if client is None:
            return ImageResponse(error=f'No credentials configured for {LABEL}')
        model = normalize_model(req.model)
        image_size, aspect_ratio = get_size_buckets(req.width, req.height)
        image_config = build_image_config(model, image_size, aspect_ratio)
        log.debug(f'Cloud: provider={PROVIDER_ID} model="{model}" size={image_size} ar={aspect_ratio} init={req.init_image is not None}')
        try:
            t0 = time.time()
            response = call_model(client, model, req.prompt, image_config, image=req.init_image)
            t1 = time.time()
            tokens = getattr(getattr(response, 'usage_metadata', None), 'total_token_count', 0)
            log.debug(f'Cloud: provider={PROVIDER_ID} model="{model}" tokens={tokens} time={(t1 - t0):.2f}')
        except Exception as e:
            log.error(f'Cloud: provider={PROVIDER_ID} model="{model}" {e}')
            return ImageResponse(error=str(e), model=model)
        images = extract_images(response)
        if not images:
            return ImageResponse(error='no images in response', model=model)
        return ImageResponse(images=images, model=model)

    return await asyncio.to_thread(run)


class GoogleNanoBananaPipeline:
    """Sync wrapper preserving the ``shared.sd_model(**base_args)`` contract.

    Submits a Job, polls JOBS until terminal, returns the legacy single-PIL-image
    payload that ``processing_diffusers.py`` expects.
    """

    def __init__(self, model_name: str):
        self.model = model_name
        log.debug(f'Load model: type=GoogleNanoBanana model="{model_name}"')

    def __call__(self, prompt, width: int, height: int, image: 'PILImage' = None):
        from modules.cloud import jobs  # pylint: disable=import-outside-toplevel
        from modules import shared  # pylint: disable=import-outside-toplevel
        text = prompt[0] if isinstance(prompt, (list, tuple)) and prompt else (prompt or '')
        req = ImageRequest(model=self.model, prompt=text, width=width, height=height, init_image=image)
        job = jobs.submit_job('image', PROVIDER_ID, req)
        watchdog = float(getattr(shared.opts, 'cloud_job_max_duration', 600.0)) + 30.0
        deadline = time.time() + watchdog
        while time.time() < deadline:
            current = jobs.get_job(job.id)
            if current is None:
                return None
            if current.status in TERMINAL_JOB_STATUSES:
                if current.status == 'succeeded' and current.result is not None and current.result.images:
                    return current.result.images[0]
                if current.error:
                    log.warning(f'Cloud: provider={PROVIDER_ID} job={job.id} error={current.error}')
                return None
            time.sleep(2)
        log.warning(f'Cloud: provider={PROVIDER_ID} job={job.id} watchdog timed out — cancelling')
        jobs.cancel_job(job.id)
        return None


def build_pipeline(model_name: str) -> GoogleNanoBananaPipeline:
    return GoogleNanoBananaPipeline(model_name)


assert GoogleNanoBananaPipeline.__name__ == SENTINEL_CLASS_NAME, (
    'GoogleNanoBananaPipeline class name drift would break literal-name dispatch in '
    'sd_models.py / modeldata.py / processing_args.py'
)


register_image(
    PROVIDER_ID,
    mode='sync',
    predict=predict_image,
    label=LABEL,
    models=list_default_models,
    enabled=is_enabled,
)
