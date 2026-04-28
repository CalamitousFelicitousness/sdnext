from __future__ import annotations
from typing import Iterator
from modules.cloud.types import (
    TextRequest,
    VisionRequest,
    TextResponse,
    VisionResponse,
    ImageRequest,
    ImageResponse,
    VideoRequest,
    VideoResponse,
    Job,
    JobStatus,
    TERMINAL_JOB_STATUSES,
    CloudError,
)
from modules.cloud.registry import (
    TEXT_PROVIDERS,
    VISION_PROVIDERS,
    IMAGE_PROVIDERS,
    VIDEO_PROVIDERS,
    REGISTRIES,
    register,
    register_text,
    register_vision,
    register_image,
    register_video,
    get_handler,
    list_providers,
    list_models,
)
from modules.cloud.jobs import (
    submit_job,
    get_job,
    list_jobs,
    cancel_job,
    JOBS,
    HISTORY,
)


def predict_text(provider_id: str, req: TextRequest) -> TextResponse:
    entry = get_handler('text', provider_id)
    if entry is None:
        return TextResponse(error=f'Unknown text provider: {provider_id}')
    try:
        return entry['predict'](req)
    except Exception as e:
        return TextResponse(error=str(e))


def predict_vision(provider_id: str, req: VisionRequest) -> VisionResponse:
    entry = get_handler('vision', provider_id)
    if entry is None:
        return VisionResponse(error=f'Unknown vision provider: {provider_id}')
    try:
        return entry['predict'](req)
    except Exception as e:
        return VisionResponse(error=str(e))


def stream_text(provider_id: str, req: TextRequest) -> Iterator[str]:
    entry = get_handler('text', provider_id)
    if entry is None:
        raise CloudError(f'Unknown text provider: {provider_id}')
    streamer = entry.get('stream')
    if streamer is None:
        resp = entry['predict'](req)
        if resp.error:
            raise CloudError(resp.error)
        yield resp.text
        return
    yield from streamer(req)


__all__ = [
    'TextRequest',
    'VisionRequest',
    'TextResponse',
    'VisionResponse',
    'ImageRequest',
    'ImageResponse',
    'VideoRequest',
    'VideoResponse',
    'Job',
    'JobStatus',
    'TERMINAL_JOB_STATUSES',
    'CloudError',
    'TEXT_PROVIDERS',
    'VISION_PROVIDERS',
    'IMAGE_PROVIDERS',
    'VIDEO_PROVIDERS',
    'REGISTRIES',
    'register',
    'register_text',
    'register_vision',
    'register_image',
    'register_video',
    'get_handler',
    'list_providers',
    'list_models',
    'predict_text',
    'predict_vision',
    'stream_text',
    'submit_job',
    'get_job',
    'list_jobs',
    'cancel_job',
    'JOBS',
    'HISTORY',
]


# side-effect imports: providers self-register on first import of modules.cloud
from modules.cloud import openai_compat  # pylint: disable=wrong-import-position
from modules.cloud import anthropic  # pylint: disable=wrong-import-position
from modules.cloud import google  # pylint: disable=wrong-import-position
from modules.cloud import google_image  # pylint: disable=wrong-import-position
from modules.cloud import google_video  # pylint: disable=wrong-import-position
