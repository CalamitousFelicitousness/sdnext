"""Shared helpers for NanoGPT image and video providers.

Both registrations lazily fetch the live model catalog from NanoGPT's public
discovery endpoints (no auth required):

  - GET https://nano-gpt.com/api/v1/image-models
  - GET https://nano-gpt.com/api/v1/video-models

Response shape: ``{ "object": "list", "data": [{ "id": "...", ... }] }``.

Results are cached for one hour. On fetch failure (offline, 5xx, timeout)
the cached list is returned if any; otherwise the curated fallback is used.
This means the dropdown stays populated even when the framework cannot reach
NanoGPT.

Filtering: NanoGPT exposes specialty pipelines (avatar, lipsync, face-swap,
longstories) under the same ``video`` category as ordinary T2V/I2V models.
Those models accept inputs that do not map onto the framework's
``VideoRequest`` shape (audio sources, character arrays, scripts) and would
just 400 if invoked from the standard dropdown. We strip them in
``filter_utility_only`` so the default selection only shows pipelines that
the standard request envelope can drive. Upscalers are kept because they
take the same ``image``/``video`` inputs as I2V.
"""
from __future__ import annotations
import threading
import time
from typing import Optional
import requests
from modules.logger import log


V1_BASE = 'https://nano-gpt.com/api/v1'
API_BASE = 'https://nano-gpt.com/api'

DISCOVERY_CACHE_TTL = 3600.0
DISCOVERY_TIMEOUT = 5.0


SKIP_PATTERNS = (
    'avatar',
    'lipsync',
    'omni-human',
    'face-swap',
    'longstories',
    'latentsync',
    'magihuman',
)


# Curated fallback lists — used when the discovery endpoint is unreachable on first call.
# Kept short and focused on flagship models per family so the UI is usable offline.

IMAGE_FALLBACK_MODELS = (
    'nano-banana-2',
    'nano-banana',
    'gpt-image-1.5',
    'gpt-image-1',
    'seedream-v4.5',
    'seedream-v4',
    'hunyuan-image-3',
    'hidream',
    'hidream-edit',
    'flux-2-pro',
    'flux-2-dev',
    'flux-kontext',
    'flux-lora/inpainting',
    'gemini-flash-edit',
    'z-image-turbo',
    'qwen-image-2.0-pro',
    'imagen-4',
    'recraft-v4',
    'midjourney',
    'Upscaler',
    'seedvr2-image',
    'clarity-ai-crystal-upscaler',
)


VIDEO_FALLBACK_MODELS = (
    # Veo
    'veo3-1-video',
    'veo3-fast-video',
    'veo3-1-lite-video',
    'veo3-video',
    'veo2-video',
    # Sora
    'sora-2',
    # Kling
    'kling-v30-pro',
    'kling-v30-std',
    'kling-v26-pro',
    'kling-v25-turbo-pro',
    'kling-video-v2',
    'kling-v21-pro',
    # Wan
    'wan-2.7-video',
    'wan-wavespeed-26',
    'wan-video-22',
    'wan-video-22-turbo',
    'wan-video-image-to-video',
    # MiniMax / Hailuo
    'minimax-hailuo-23-pro',
    'minimax-hailuo-02-pro',
    # Hunyuan
    'hunyuan-video-15',
    'hunyuan-video-image-to-video',
    # Bytedance Seedance
    'bytedance-seedance-2-0',
    'bytedance-seedance-2-0-fast',
    'seedance-video',
    'seedance-lite-video',
    # Pixverse
    'pixverse-v6',
    'pixverse-v55',
    # Lightricks
    'lightricks-ltx-2-pro',
    'lightricks-ltx-2-fast',
    # Vidu
    'vidu-q3',
    'vidu-video',
    # Runway
    'runwayml-gen4-aleph',
    'runway-gen-45',
    # Other
    'happyhorse-1.0',
    'kandinsky5-pro',
    # Utility (kept per "core utility" curation)
    'video-upscaler',
    'seedvr2-video-upscaler',
    'bytedance-seedance-upscaler',
    'wavespeed-ai/music-video-generator',
)


CACHE: dict = {
    'image': {'models': None, 'resolutions': {}, 'ts': 0.0},
    'video': {'models': None, 'ts': 0.0},
}
CACHE_LOCK = threading.Lock()


def is_enabled() -> bool:
    from modules import shared  # pylint: disable=import-outside-toplevel
    return bool(getattr(shared.opts, 'nanogpt_key', ''))


def auth_headers() -> dict:
    from modules import shared  # pylint: disable=import-outside-toplevel
    key = getattr(shared.opts, 'nanogpt_key', '') or ''
    return {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}


def filter_utility_only(model_ids: list[str]) -> list[str]:
    """Drop avatar / lipsync / face-swap / longstories pipelines."""
    out: list[str] = []
    for mid in model_ids:
        if not mid:
            continue
        low = mid.lower()
        if any(pat in low for pat in SKIP_PATTERNS):
            continue
        out.append(mid)
    return out


def fetch_models(capability: str) -> Optional[tuple[list[str], dict[str, list[str]]]]:
    """Fetch live model IDs and per-model resolutions from NanoGPT discovery.

    Returns ``(ids, resolutions_map)`` on success, ``None`` on failure.
    ``resolutions_map`` maps model id → list of accepted resolution strings
    (only populated when the model entry includes ``supported_parameters.resolutions``).
    """
    url = f'{V1_BASE}/{capability}-models'
    try:
        resp = requests.get(url, timeout=DISCOVERY_TIMEOUT)
        if resp.status_code >= 400:
            log.debug(f'NanoGPT discovery: {url} HTTP {resp.status_code}')
            return None
        payload = resp.json()
    except Exception as e:
        log.debug(f'NanoGPT discovery: {url} {e}')
        return None
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, list):
        log.debug(f'NanoGPT discovery: {url} unexpected payload shape')
        return None
    ids: list[str] = []
    resolutions: dict[str, list[str]] = {}
    for entry in data:
        if not isinstance(entry, dict) or not entry.get('id'):
            continue
        mid = str(entry['id'])
        ids.append(mid)
        params = entry.get('supported_parameters') or {}
        res_list = params.get('resolutions')
        if isinstance(res_list, list) and res_list:
            resolutions[mid] = [str(r) for r in res_list]
    return ids, resolutions


def list_models_cached(capability: str, fallback: tuple[str, ...]) -> list[str]:
    """Return the cached live list when fresh; refresh on miss with fallback on failure."""
    now = time.time()
    with CACHE_LOCK:
        cached = CACHE.get(capability) or {}
        models = cached.get('models')
        ts = cached.get('ts') or 0.0
        if models is not None and (now - ts) < DISCOVERY_CACHE_TTL:
            return list(models)
    result = fetch_models(capability)
    if result is not None:
        ids, resolutions = result
        filtered = filter_utility_only(ids)
        with CACHE_LOCK:
            CACHE[capability]['models'] = filtered
            CACHE[capability]['ts'] = now
            if resolutions:
                CACHE[capability]['resolutions'] = resolutions
        return list(filtered)
    with CACHE_LOCK:
        stale = (CACHE.get(capability) or {}).get('models')
    if stale:
        return list(stale)
    return list(fallback)


def get_model_resolutions(model_id: str) -> list[str]:
    """Return supported resolution strings for a NanoGPT image model from the discovery cache.

    Triggers a cache population if the resolution map is empty (first call
    before any list_models_cached invocation).
    """
    with CACHE_LOCK:
        resolutions = (CACHE.get('image', {}).get('resolutions') or {})
        if resolutions:
            return list(resolutions.get(model_id, []))
    list_models_cached('image', IMAGE_FALLBACK_MODELS)
    with CACHE_LOCK:
        resolutions = (CACHE.get('image', {}).get('resolutions') or {})
        return list(resolutions.get(model_id, []))


def reset_cache() -> None:
    """Wipe the discovery cache."""
    with CACHE_LOCK:
        for cap in CACHE:
            CACHE[cap]['models'] = None
            CACHE[cap]['ts'] = 0.0
            if 'resolutions' in CACHE[cap]:
                CACHE[cap]['resolutions'] = {}
