from __future__ import annotations
import os
from typing import Iterator, Optional
from modules.logger import log
from modules.cloud.types import TextRequest, VisionRequest, TextResponse, VisionResponse
from modules.cloud.registry import register_text, register_vision
from modules.cloud.client import post_json, stream_sse, image_to_base64, mask_key


PROVIDER_ID = 'anthropic'
LABEL = 'Anthropic'
BASE_URL = 'https://api.anthropic.com/v1'
MESSAGES_PATH = '/messages'
API_VERSION = '2023-06-01'
KEY_OPT = 'anthropic_key'
ENV_VAR = 'ANTHROPIC_API_KEY'
DEFAULT_MAX_TOKENS = 4096
THINKING_BUDGET_TOKENS = 8192
IMAGE_MAX_DIM = 2048

DEFAULT_MODELS = (
    'claude-opus-4-7',
    'claude-sonnet-4-6',
    'claude-haiku-4-5-20251001',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-haiku-20241022',
)


def resolve_key() -> str:
    from modules import shared  # pylint: disable=import-outside-toplevel
    val = getattr(shared.opts, KEY_OPT, '') or ''
    if val:
        return val
    return os.environ.get(ENV_VAR, '') or ''


def is_enabled() -> bool:
    return bool(resolve_key())


def list_default_models() -> list[str]:
    return list(DEFAULT_MODELS)


def get_request_timeout() -> int:
    from modules import shared  # pylint: disable=import-outside-toplevel
    try:
        return int(getattr(shared.opts, 'cloud_request_timeout', 60))
    except (TypeError, ValueError):
        return 60


def build_headers(key: str) -> dict:
    return {
        'x-api-key': key,
        'anthropic-version': API_VERSION,
        'content-type': 'application/json',
    }


def build_messages(req: TextRequest, *, image_b64: Optional[str] = None) -> list[dict]:
    user_text = req.prompt or ''
    user_content: list[dict]
    if image_b64:
        user_content = [
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/jpeg',
                    'data': image_b64,
                },
            },
            {'type': 'text', 'text': user_text},
        ]
    else:
        user_content = [{'type': 'text', 'text': user_text}]
    messages: list[dict] = [{'role': 'user', 'content': user_content}]
    if req.prefill:
        messages.append({'role': 'assistant', 'content': req.prefill})
    return messages


def build_body(req: TextRequest, messages: list[dict], *, stream: bool = False) -> dict:
    body: dict = {
        'model': req.model,
        'messages': messages,
        'max_tokens': req.max_tokens or DEFAULT_MAX_TOKENS,
        'stream': stream,
    }
    if req.system:
        body['system'] = req.system
    if req.temperature is not None:
        body['temperature'] = req.temperature
    if req.top_p is not None:
        body['top_p'] = req.top_p
    if req.top_k is not None:
        body['top_k'] = req.top_k
    if req.thinking:
        body['thinking'] = {'type': 'enabled', 'budget_tokens': THINKING_BUDGET_TOKENS}
    if req.extra:
        for k, v in req.extra.items():
            if k not in body:
                body[k] = v
    return body


def extract_text_blocks(content: list) -> str:
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get('type') == 'text':
            parts.append(block.get('text', ''))
    return ''.join(parts)


def anthropic_event_extractor(evt: dict) -> Optional[str]:
    if not isinstance(evt, dict):
        return None
    if evt.get('type') != 'content_block_delta':
        return None
    delta = evt.get('delta') or {}
    if delta.get('type') != 'text_delta':
        return None
    return delta.get('text') or None


def predict_text(req: TextRequest) -> TextResponse:
    key = resolve_key()
    if not key:
        return TextResponse(error=f'No API key configured for {LABEL}')
    url = f'{BASE_URL.rstrip("/")}{MESSAGES_PATH}'
    headers = build_headers(key)
    messages = build_messages(req)
    body = build_body(req, messages, stream=False)
    log.debug(f'Cloud: provider={PROVIDER_ID} model={req.model} url={url} key={mask_key(key)}')
    try:
        result = post_json(url, headers, body, timeout=get_request_timeout())
    except Exception as e:
        return TextResponse(error=str(e))
    text = extract_text_blocks(result.get('content', []))
    return TextResponse(
        text=text,
        finish_reason=result.get('stop_reason'),
        usage=result.get('usage'),
        model=result.get('model') or req.model,
    )


def predict_vision(req: VisionRequest) -> VisionResponse:
    key = resolve_key()
    if not key:
        return VisionResponse(error=f'No API key configured for {LABEL}')
    image_b64 = None
    if req.image is not None:
        image_b64 = image_to_base64(req.image, max_dim=IMAGE_MAX_DIM)
    url = f'{BASE_URL.rstrip("/")}{MESSAGES_PATH}'
    headers = build_headers(key)
    messages = build_messages(req, image_b64=image_b64)
    body = build_body(req, messages, stream=False)
    log.debug(f'Cloud: provider={PROVIDER_ID} model={req.model} url={url} key={mask_key(key)} vision=true')
    try:
        result = post_json(url, headers, body, timeout=get_request_timeout())
    except Exception as e:
        return VisionResponse(error=str(e))
    text = extract_text_blocks(result.get('content', []))
    return VisionResponse(
        text=text,
        finish_reason=result.get('stop_reason'),
        usage=result.get('usage'),
        model=result.get('model') or req.model,
    )


def stream_text(req: TextRequest) -> Iterator[str]:
    key = resolve_key()
    if not key:
        raise RuntimeError(f'No API key configured for {LABEL}')
    url = f'{BASE_URL.rstrip("/")}{MESSAGES_PATH}'
    headers = build_headers(key)
    messages = build_messages(req)
    body = build_body(req, messages, stream=True)
    log.debug(f'Cloud: stream provider={PROVIDER_ID} model={req.model} url={url} key={mask_key(key)}')
    yield from stream_sse(url, headers, body, event_extractor=anthropic_event_extractor)


register_text(
    PROVIDER_ID,
    predict=predict_text,
    stream=stream_text,
    label=LABEL,
    models=list_default_models,
    enabled=is_enabled,
)

register_vision(
    PROVIDER_ID,
    predict=predict_vision,
    label=LABEL,
    models=list_default_models,
    enabled=is_enabled,
)
