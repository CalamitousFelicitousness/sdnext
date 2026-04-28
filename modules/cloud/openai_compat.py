from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional
from modules.logger import log
from modules.cloud.types import TextRequest, VisionRequest, TextResponse, VisionResponse
from modules.cloud.registry import register_text, register_vision
from modules.cloud.client import post_json, stream_sse, image_to_data_url, mask_key




@dataclass
class Preset:
    provider_id: str
    label: str
    base_url: str
    key_opt: str
    env_var: str = ''
    extra_headers: dict = field(default_factory=dict)
    chat_path: str = '/chat/completions'
    models_path: str = '/models'
    default_models: tuple = ()
    supports_vision: bool = True
    supports_streaming: bool = True
    image_max_dim: int = 2048


PRESETS: dict[str, Preset] = {
    'openai': Preset(
        provider_id='openai',
        label='OpenAI',
        base_url='https://api.openai.com/v1',
        key_opt='openai_key',
        env_var='OPENAI_API_KEY',
        default_models=(
            'gpt-4o-mini',
            'gpt-4o',
            'gpt-4.1-mini',
            'gpt-4.1',
            'o4-mini',
            'o3-mini',
        ),
    ),
    'openrouter': Preset(
        provider_id='openrouter',
        label='OpenRouter',
        base_url='https://openrouter.ai/api/v1',
        key_opt='openrouter_key',
        env_var='OPENROUTER_API_KEY',
        extra_headers={
            'HTTP-Referer': 'https://github.com/vladmandic/sdnext',
            'X-Title': 'SD.Next',
        },
        default_models=(
            'openai/gpt-4o-mini',
            'anthropic/claude-3.5-sonnet',
            'anthropic/claude-3.5-haiku',
            'google/gemini-2.5-flash',
            'meta-llama/llama-3.3-70b-instruct',
            'qwen/qwen-2.5-72b-instruct',
        ),
    ),
    'nanogpt': Preset(
        provider_id='nanogpt',
        label='NanoGPT',
        base_url='https://nano-gpt.com/api/v1',
        key_opt='nanogpt_key',
        env_var='NANOGPT_API_KEY',
        default_models=(
            'chatgpt-4o-latest',
            'claude-3-5-sonnet',
            'gemini-2.5-flash',
        ),
    ),
    'aihubmix': Preset(
        provider_id='aihubmix',
        label='AIHubMix',
        base_url='https://aihubmix.com/v1',
        key_opt='aihubmix_key',
        env_var='AIHUBMIX_API_KEY',
        default_models=(
            'gpt-4o-mini',
            'claude-3-5-sonnet',
            'gemini-2.5-pro',
        ),
    ),
    'huggingface': Preset(
        provider_id='huggingface',
        label='HuggingFace Inference Providers',
        base_url='https://router.huggingface.co/v1',
        key_opt='huggingface_token',
        env_var='HF_TOKEN',
        default_models=(
            'meta-llama/Llama-3.3-70B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            'mistralai/Mistral-Small-Instruct-2409',
        ),
        supports_vision=False,
    ),
    'openai_compat_custom': Preset(
        provider_id='openai_compat_custom',
        label='Custom OpenAI-compatible',
        base_url='',
        key_opt='openai_compat_custom_key',
        env_var='',
        default_models=(),
    ),
}


def resolve_key(preset: Preset) -> str:
    from modules import shared  # pylint: disable=import-outside-toplevel
    val = getattr(shared.opts, preset.key_opt, '') or ''
    if val:
        return val
    if preset.env_var:
        return os.environ.get(preset.env_var, '') or ''
    return ''


def resolve_base_url(preset: Preset) -> str:
    if preset.provider_id == 'openai_compat_custom':
        from modules import shared  # pylint: disable=import-outside-toplevel
        return getattr(shared.opts, 'openai_compat_custom_url', '') or ''
    if preset.provider_id == 'openai':
        from modules import shared  # pylint: disable=import-outside-toplevel
        override = getattr(shared.opts, 'openai_base_override', '') or ''
        if override:
            return override
    return preset.base_url


def resolve_models(preset: Preset) -> list[str]:
    if preset.provider_id == 'openai_compat_custom':
        from modules import shared  # pylint: disable=import-outside-toplevel
        raw = getattr(shared.opts, 'openai_compat_custom_models', '') or ''
        return [m.strip() for m in raw.split(',') if m.strip()]
    return list(preset.default_models)


def is_enabled(preset: Preset) -> bool:
    if preset.provider_id == 'openai_compat_custom':
        return bool(resolve_key(preset)) and bool(resolve_base_url(preset))
    return bool(resolve_key(preset))


def build_headers(preset: Preset, key: str) -> dict:
    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
    }
    headers.update(preset.extra_headers)
    return headers


def build_messages(req: TextRequest, *, image_data_url: Optional[str] = None) -> list[dict]:
    messages: list[dict] = []
    if req.system:
        messages.append({'role': 'system', 'content': req.system})
    user_text = req.prompt or ''
    if req.prefill:
        user_text = f'{user_text}{req.prefill}'
    if image_data_url:
        user_content = [
            {'type': 'text', 'text': user_text},
            {'type': 'image_url', 'image_url': {'url': image_data_url}},
        ]
        messages.append({'role': 'user', 'content': user_content})
    else:
        messages.append({'role': 'user', 'content': user_text})
    return messages


def build_body(req: TextRequest, model: str, messages: list[dict], *, stream: bool = False) -> dict:
    body: dict = {
        'model': model,
        'messages': messages,
        'stream': stream,
    }
    if req.temperature is not None:
        body['temperature'] = req.temperature
    if req.max_tokens is not None:
        body['max_tokens'] = req.max_tokens
    if req.top_p is not None:
        body['top_p'] = req.top_p
    if req.extra:
        for k, v in req.extra.items():
            if k not in body:
                body[k] = v
    return body


def get_request_timeout() -> int:
    from modules import shared  # pylint: disable=import-outside-toplevel
    try:
        return int(getattr(shared.opts, 'cloud_request_timeout', 60))
    except (TypeError, ValueError):
        return 60


def predict_text_for(preset_id: str, req: TextRequest) -> TextResponse:
    preset = PRESETS.get(preset_id)
    if preset is None:
        return TextResponse(error=f'Unknown preset: {preset_id}')
    key = resolve_key(preset)
    if not key:
        return TextResponse(error=f'No API key configured for {preset.label}')
    base_url = resolve_base_url(preset)
    if not base_url:
        return TextResponse(error=f'No base URL configured for {preset.label}')
    url = f'{base_url.rstrip("/")}{preset.chat_path}'
    headers = build_headers(preset, key)
    messages = build_messages(req)
    body = build_body(req, req.model, messages, stream=False)
    log.debug(f'Cloud: provider={preset_id} model={req.model} url={url} key={mask_key(key)}')
    try:
        result = post_json(url, headers, body, timeout=get_request_timeout())
    except Exception as e:
        return TextResponse(error=str(e))
    try:
        choice = result['choices'][0]
        text = choice.get('message', {}).get('content', '') or ''
        return TextResponse(
            text=text,
            finish_reason=choice.get('finish_reason'),
            usage=result.get('usage'),
            model=result.get('model') or req.model,
        )
    except (KeyError, IndexError, TypeError) as e:
        return TextResponse(error=f'Malformed response: {e}', raw=result)


def predict_vision_for(preset_id: str, req: VisionRequest) -> VisionResponse:
    preset = PRESETS.get(preset_id)
    if preset is None:
        return VisionResponse(error=f'Unknown preset: {preset_id}')
    if not preset.supports_vision:
        return VisionResponse(error=f'{preset.label} does not support vision')
    key = resolve_key(preset)
    if not key:
        return VisionResponse(error=f'No API key configured for {preset.label}')
    base_url = resolve_base_url(preset)
    if not base_url:
        return VisionResponse(error=f'No base URL configured for {preset.label}')
    image_url = None
    if req.image is not None:
        image_url = image_to_data_url(req.image, max_dim=preset.image_max_dim)
    url = f'{base_url.rstrip("/")}{preset.chat_path}'
    headers = build_headers(preset, key)
    messages = build_messages(req, image_data_url=image_url)
    body = build_body(req, req.model, messages, stream=False)
    log.debug(f'Cloud: provider={preset_id} model={req.model} url={url} key={mask_key(key)} vision=true')
    try:
        result = post_json(url, headers, body, timeout=get_request_timeout())
    except Exception as e:
        return VisionResponse(error=str(e))
    try:
        choice = result['choices'][0]
        text = choice.get('message', {}).get('content', '') or ''
        return VisionResponse(
            text=text,
            finish_reason=choice.get('finish_reason'),
            usage=result.get('usage'),
            model=result.get('model') or req.model,
        )
    except (KeyError, IndexError, TypeError) as e:
        return VisionResponse(error=f'Malformed response: {e}', raw=result)


def openai_event_extractor(evt: dict) -> Optional[str]:
    try:
        choices = evt.get('choices') or []
        if not choices:
            return None
        delta = choices[0].get('delta') or {}
        return delta.get('content') or None
    except (AttributeError, TypeError):
        return None


def stream_text_for(preset_id: str, req: TextRequest) -> Iterator[str]:
    preset = PRESETS.get(preset_id)
    if preset is None:
        raise RuntimeError(f'Unknown preset: {preset_id}')
    if not preset.supports_streaming:
        result = predict_text_for(preset_id, req)
        if result.error:
            raise RuntimeError(result.error)
        yield result.text
        return
    key = resolve_key(preset)
    if not key:
        raise RuntimeError(f'No API key configured for {preset.label}')
    base_url = resolve_base_url(preset)
    if not base_url:
        raise RuntimeError(f'No base URL configured for {preset.label}')
    url = f'{base_url.rstrip("/")}{preset.chat_path}'
    headers = build_headers(preset, key)
    messages = build_messages(req)
    body = build_body(req, req.model, messages, stream=True)
    log.debug(f'Cloud: stream provider={preset_id} model={req.model} url={url} key={mask_key(key)}')
    yield from stream_sse(url, headers, body, event_extractor=openai_event_extractor)


def make_predict_text(preset_id: str):
    def predict(req: TextRequest) -> TextResponse:
        return predict_text_for(preset_id, req)
    return predict


def make_predict_vision(preset_id: str):
    def predict(req: VisionRequest) -> VisionResponse:
        return predict_vision_for(preset_id, req)
    return predict


def make_stream_text(preset_id: str):
    def stream(req: TextRequest) -> Iterator[str]:
        yield from stream_text_for(preset_id, req)
    return stream


def make_models_callable(preset_id: str):
    def models() -> list[str]:
        preset = PRESETS.get(preset_id)
        if preset is None:
            return []
        return resolve_models(preset)
    return models


def make_enabled_callable(preset_id: str):
    def enabled() -> bool:
        preset = PRESETS.get(preset_id)
        if preset is None:
            return False
        return is_enabled(preset)
    return enabled


def register_all() -> None:
    for preset_id, preset in PRESETS.items():
        register_text(
            preset_id,
            predict=make_predict_text(preset_id),
            stream=make_stream_text(preset_id) if preset.supports_streaming else None,
            label=preset.label,
            models=make_models_callable(preset_id),
            enabled=make_enabled_callable(preset_id),
        )
        if preset.supports_vision:
            register_vision(
                preset_id,
                predict=make_predict_vision(preset_id),
                label=preset.label,
                models=make_models_callable(preset_id),
                enabled=make_enabled_callable(preset_id),
            )


register_all()
