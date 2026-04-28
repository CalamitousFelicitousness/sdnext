from __future__ import annotations
import io
import os
from typing import Iterator, Optional
from modules.logger import log
from modules.cloud.types import TextRequest, VisionRequest, TextResponse, VisionResponse
from modules.cloud.registry import register_text, register_vision
from modules.cloud.client import mask_key


PROVIDER_ID = 'google'
LABEL = 'Google'
PACKAGE_PIN = 'google-genai==1.52.0'

DEFAULT_MODELS = (
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-3.1-pro-preview',
    'gemini-3.1-flash-lite-preview',
    'gemini-3-flash-preview',
)


debug_enabled = os.environ.get('SD_CAPTION_DEBUG', None) is not None
debug_log = log.trace if debug_enabled else (lambda *args, **kwargs: None)


client_state: dict = {'instance': None, 'args_key': None}


def resolve_args() -> Optional[dict]:
    from modules.shared import opts  # pylint: disable=import-outside-toplevel
    api_key = opts.google_api_key or ''
    project_id = opts.google_project_id or ''
    location_id = opts.google_location_id or ''
    use_vertexai = bool(opts.google_use_vertexai)

    has_api_key = bool(api_key)
    has_project = bool(project_id)
    has_location = bool(location_id)

    # No credentials set at all -> silently disabled (registry-probe path).
    if not (has_api_key or has_project or has_location):
        return None

    if use_vertexai:
        if has_api_key and (has_project or has_location):
            log.error('Cloud: google API key and project/location are mutually exclusive')
            return None
        if has_api_key:
            return {'api_key': api_key, 'vertexai': True}
        if has_project and has_location:
            return {'vertexai': True, 'project': project_id, 'location': location_id}
        log.error('Cloud: google Vertex AI requires API key (Express Mode) or project ID + location ID')
        return None
    if not has_api_key:
        return None
    return {'api_key': api_key}


def is_enabled() -> bool:
    return resolve_args() is not None


def list_default_models() -> list[str]:
    return list(DEFAULT_MODELS)


def reset_client() -> None:
    client_state['instance'] = None
    client_state['args_key'] = None


def normalize_model(model: str) -> str:
    if not model:
        return model
    if model.startswith('google/'):
        return model[len('google/'):]
    return model


def get_client():
    args = resolve_args()
    if args is None:
        return None
    args_key = tuple(sorted(args.items()))
    if client_state['instance'] is None or client_state['args_key'] != args_key:
        from installer import install  # pylint: disable=import-outside-toplevel
        install(PACKAGE_PIN)
        from google import genai  # pylint: disable=import-outside-toplevel,no-name-in-module
        log_args = {**args}
        if 'api_key' in log_args:
            log_args['api_key'] = mask_key(log_args['api_key'])
        log.debug(f'Cloud: provider={PROVIDER_ID} args={log_args}')
        client_state['instance'] = genai.Client(**args)
        client_state['args_key'] = args_key
    return client_state['instance']


def build_config(req: TextRequest):
    from google.genai import types  # pylint: disable=import-outside-toplevel,no-name-in-module
    from modules import shared  # pylint: disable=import-outside-toplevel
    config: dict = {
        'system_instruction': req.system or shared.opts.caption_vlm_system,
        'thinking_config': types.ThinkingConfig(thinking_level='high' if req.thinking else 'low'),
    }
    if req.temperature is not None:
        config['temperature'] = req.temperature
    if req.max_tokens is not None:
        config['max_output_tokens'] = req.max_tokens
    if req.top_p is not None:
        config['top_p'] = req.top_p
    if req.top_k is not None:
        config['top_k'] = req.top_k
    return config


def build_contents(prompt: str, prefill: Optional[str], image=None) -> list:
    from google.genai import types  # pylint: disable=import-outside-toplevel,no-name-in-module
    cleaned = (prompt or '').replace('<', '').replace('>', '').replace('_', ' ')
    if prefill:
        cleaned += prefill
    if image is not None:
        buf = io.BytesIO()
        save_img = image
        if image.mode != 'RGB':
            save_img = image.convert('RGB')
        save_img.save(buf, format='JPEG')
        return [types.Part.from_bytes(data=buf.getvalue(), mime_type='image/jpeg'), cleaned]
    return [cleaned]


def predict_text(req: TextRequest) -> TextResponse:
    client = get_client()
    if client is None:
        return TextResponse(error=f'No credentials configured for {LABEL}')
    model = normalize_model(req.model)
    config = build_config(req)
    contents = build_contents(req.prompt, req.prefill)
    debug_log(f'Gemini config: {config}')
    debug_log(f'Gemini text contents: {contents}')
    try:
        response = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as e:
        log.error(f'Cloud: provider={PROVIDER_ID} error={e}')
        return TextResponse(error=str(e))
    return TextResponse(text=response.text or '', model=model)


def predict_vision(req: VisionRequest) -> VisionResponse:
    client = get_client()
    if client is None:
        return VisionResponse(error=f'No credentials configured for {LABEL}')
    model = normalize_model(req.model)
    config = build_config(req)
    contents = build_contents(req.prompt, req.prefill, image=req.image)
    debug_log(f'Gemini config: {config}')
    try:
        response = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as e:
        log.error(f'Cloud: provider={PROVIDER_ID} error={e}')
        return VisionResponse(error=str(e))
    return VisionResponse(text=response.text or '', model=model)


def stream_text(req: TextRequest) -> Iterator[str]:
    client = get_client()
    if client is None:
        raise RuntimeError(f'No credentials configured for {LABEL}')
    model = normalize_model(req.model)
    config = build_config(req)
    contents = build_contents(req.prompt, req.prefill)
    debug_log(f'Gemini stream config: {config}')
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        text = getattr(chunk, 'text', None)
        if text:
            yield text


def predict(question, image, vqa_model, system_prompt, model_name, prefill, thinking, gen_kwargs):  # pylint: disable=unused-argument
    """Back-compat wrapper preserving the legacy modules/caption/gemini.py:predict() signature."""
    req = VisionRequest(
        model=vqa_model,
        prompt=question,
        system=system_prompt,
        prefill=prefill,
        thinking=bool(thinking),
        image=image,
        temperature=gen_kwargs.get('temperature') if gen_kwargs else None,
        max_tokens=gen_kwargs.get('max_output_tokens') if gen_kwargs else None,
    )
    resp = predict_vision(req)
    return resp.text if not resp.error else f'Error: {resp.error}'


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
