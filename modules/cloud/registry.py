"""Cloud provider registry.

Each capability registry maps `provider_id -> handler dict`. Handler dict shape:

    {
        'mode':          'sync' | 'async',
        'predict':       async or sync callable (sync mode only),
        'stream':        async or sync iterator factory (text streaming, optional),
        'submit':        async callable taking Job, mutating job.extra (async mode),
        'poll':          async callable taking Job, mutating job.status/result (async mode),
        'cancel':        async callable taking Job; best-effort, optional,
        'poll_interval': float seconds, defaults to opts.cloud_job_poll_default,
        'label':         human-readable string,
        'models':        list[str] | callable returning list[str],
        'enabled':       callable returning bool,
    }

Mode contract:
- mode='sync'   requires `predict`. The runner awaits it; if it is a plain
                callable rather than a coroutine, the runner wraps the call
                with asyncio.to_thread.
- mode='async'  requires `submit` and `poll`. `cancel` is optional (best-effort).
                `predict` is forbidden in this mode.

Phase 1 text/vision providers register with `predict=` only (mode defaults to
'sync'). Phase 2 image/video providers may register either mode; long-poll
providers (Veo, Civitai, BFL) use 'async'.
"""
from __future__ import annotations
from typing import Any, Callable, Optional


TEXT_PROVIDERS: dict[str, dict] = {}
VISION_PROVIDERS: dict[str, dict] = {}
IMAGE_PROVIDERS: dict[str, dict] = {}
VIDEO_PROVIDERS: dict[str, dict] = {}


REGISTRIES: dict[str, dict[str, dict]] = {
    'text': TEXT_PROVIDERS,
    'vision': VISION_PROVIDERS,
    'image': IMAGE_PROVIDERS,
    'video': VIDEO_PROVIDERS,
}


def register(capability: str, provider_id: str, *,
             predict: Optional[Callable] = None,
             stream: Optional[Callable] = None,
             label: Optional[str] = None,
             models: Any = None,
             enabled: Optional[Callable] = None,
             mode: str = 'sync',
             submit: Optional[Callable] = None,
             poll: Optional[Callable] = None,
             cancel: Optional[Callable] = None,
             poll_interval: Optional[float] = None) -> None:
    if capability not in REGISTRIES:
        raise ValueError(f"Unknown capability: {capability!r}")
    if mode not in ('sync', 'async'):
        raise ValueError(f"register({provider_id}): mode must be 'sync' or 'async', got {mode!r}")
    if mode == 'sync':
        if predict is None:
            raise ValueError(f"register({provider_id}): mode='sync' requires predict=")
        if submit is not None or poll is not None:
            raise ValueError(f"register({provider_id}): mode='sync' forbids submit=/poll=")
    else:
        if submit is None or poll is None:
            raise ValueError(f"register({provider_id}): mode='async' requires submit= and poll=")
        if predict is not None:
            raise ValueError(f"register({provider_id}): mode='async' forbids predict=")
    REGISTRIES[capability][provider_id] = {
        'mode': mode,
        'predict': predict,
        'stream': stream,
        'submit': submit,
        'poll': poll,
        'cancel': cancel,
        'poll_interval': poll_interval,
        'label': label or provider_id,
        'models': models if models is not None else [],
        'enabled': enabled if enabled is not None else (lambda: True),
    }


def register_text(provider_id: str, **kwargs) -> None:
    register('text', provider_id, **kwargs)


def register_vision(provider_id: str, **kwargs) -> None:
    register('vision', provider_id, **kwargs)


def register_image(provider_id: str, **kwargs) -> None:
    register('image', provider_id, **kwargs)


def register_video(provider_id: str, **kwargs) -> None:
    register('video', provider_id, **kwargs)


def get_handler(capability: str, provider_id: str) -> Optional[dict]:
    if capability not in REGISTRIES:
        return None
    return REGISTRIES[capability].get(provider_id)


def list_providers(capability: str, *, only_enabled: bool = True) -> list[str]:
    if capability not in REGISTRIES:
        return []
    out = []
    for pid, entry in REGISTRIES[capability].items():
        if only_enabled:
            try:
                if not entry['enabled']():
                    continue
            except Exception:
                continue
        out.append(pid)
    return out


def list_models(capability: str, provider_id: str) -> list[str]:
    entry = get_handler(capability, provider_id)
    if entry is None:
        return []
    models = entry['models']
    if callable(models):
        try:
            return list(models())
        except Exception:
            return []
    return list(models or [])
