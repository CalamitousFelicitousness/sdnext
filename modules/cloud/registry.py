"""Cloud provider registry: opts-backed config store + adapter cache.

The adapter cache lives at module scope. CRUD operations take a
threading.Lock; reads are dict-safe and need no lock. Per-provider key
options are registered dynamically so the suffix-based secrets routing in
modules/options_handler.py works.
"""

import json
import os
import re
import threading
from dataclasses import asdict, dataclass

import gradio as gr

from modules import shared
from modules.logger import log
from modules.options import OptionInfo

from modules.cloud.adapter import OpenAICompatAdapter
from modules.cloud.errors import CloudError
from modules.cloud.presets import PRESETS
from modules.cloud.protocol import ProviderAdapter


# Env var fallbacks for built-in preset keys. Matches enso_api/cloud/config.py:13-17.
ENV_KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "nanogpt": "NANOGPT_API_KEY",
    "aihubmix": "AIHUBMIX_API_KEY",
}


@dataclass
class ProviderConfig:
    """Persistent provider config. API key is NOT a field; resolve via resolve_key()."""
    id: str
    name: str
    preset: str
    base_url: str
    enabled: bool = True


# Module-level adapter cache. CRUD ops mutate cache + opts under cache_lock.
adapters: dict[str, OpenAICompatAdapter] = {}
cache_lock = threading.Lock()


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "provider"


def make_unique_id(slug: str, existing: list[str]) -> str:
    if slug not in existing:
        return slug
    for i in range(2, 100):
        candidate = f"{slug}_{i}"
        if candidate not in existing:
            return candidate
    return f"{slug}_{len(existing)}"


def key_option_name(provider_id: str) -> str:
    return f"cloud_{provider_id}_key"


def register_key_option(provider_id: str) -> None:
    """Register the cloud_<id>_key OptionInfo so setattr/save flow works.

    Options.__setattr__ silently drops writes to keys not present in
    data_labels, data, or secrets, so this MUST run before the first key
    write.
    """
    name = key_option_name(provider_id)
    if name in shared.opts.data_labels:
        return
    info = OptionInfo(
        "",
        f"API key for cloud provider {provider_id}",
        gr.Textbox,
        {"visible": False},
    )
    info.section = ("cloud", "Cloud Providers")
    shared.opts.add_option(name, info)


def resolve_key(provider_id: str, preset_name: str) -> str:
    """Return the API key for the provider. Env var (per preset) beats stored opt."""
    env_var = ENV_KEY_MAP.get(preset_name)
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val:
            return env_val
    return getattr(shared.opts, key_option_name(provider_id), "") or ""


def read_providers() -> list[ProviderConfig]:
    """Parse the JSON-encoded cloud_providers opt into a ProviderConfig list."""
    raw = getattr(shared.opts, "cloud_providers", "[]") or "[]"
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log.warning("Cloud: cloud_providers opt is not valid JSON, treating as empty")
        return []
    if not isinstance(items, list):
        return []
    out: list[ProviderConfig] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        out.append(ProviderConfig(
            id=item["id"],
            name=item.get("name", item["id"]),
            preset=item.get("preset", "custom"),
            base_url=item.get("base_url", ""),
            enabled=item.get("enabled", True),
        ))
    return out


def write_providers(providers: list[ProviderConfig]) -> None:
    """Persist the provider list as JSON in the cloud_providers opt + save."""
    payload = json.dumps([asdict(p) for p in providers])
    shared.opts.set("cloud_providers", payload)
    shared.opts.save()


def hydrate_key_options() -> None:
    """Register key OptionInfos for every existing provider so secrets.json
    keys loaded at boot have matching data_labels entries.

    Called from modules/cloud/__init__.py at import time so subsequent
    reads/writes to cloud_<id>_key route correctly.
    """
    for cfg in read_providers():
        register_key_option(cfg.id)


def make_adapter(cfg: ProviderConfig) -> OpenAICompatAdapter:
    preset = PRESETS.get(cfg.preset, PRESETS["custom"])
    key = resolve_key(cfg.id, cfg.preset)
    adapter = OpenAICompatAdapter(cfg.id, cfg.base_url, preset, key)
    log.debug(f"Cloud: adapter created provider={cfg.id} name={cfg.name!r} preset={cfg.preset} base_url={cfg.base_url}")
    return adapter


# ---- public API ------------------------------------------------------------------

def list_providers() -> list[ProviderConfig]:
    """All configured providers, regardless of enabled status."""
    return read_providers()


def get_provider(provider_id: str) -> ProviderConfig | None:
    for cfg in read_providers():
        if cfg.id == provider_id:
            return cfg
    return None


def get_adapter(provider_id: str) -> ProviderAdapter:
    """Return cached adapter. Lazily creates on first access if needed.

    Raises ValueError if the provider isn't configured or is disabled.
    """
    cached = adapters.get(provider_id)
    if cached is not None:
        return cached
    cfg = get_provider(provider_id)
    if cfg is None:
        raise ValueError(f"Cloud provider not configured: {provider_id}")
    if not cfg.enabled:
        raise ValueError(f"Cloud provider disabled: {provider_id}")
    with cache_lock:
        cached = adapters.get(provider_id)
        if cached is None:
            cached = make_adapter(cfg)
            adapters[provider_id] = cached
    return cached


def add_provider(name: str, preset: str, base_url: str, key: str = "") -> ProviderConfig:
    """Create a new provider, persist, register its key OptionInfo, build adapter."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    with cache_lock:
        existing = read_providers()
        existing_ids = [p.id for p in existing]
        provider_id = make_unique_id(slugify(name), existing_ids)
        cfg = ProviderConfig(
            id=provider_id,
            name=name,
            preset=preset,
            base_url=base_url.rstrip("/"),
            enabled=True,
        )
        existing.append(cfg)
        write_providers(existing)
        register_key_option(provider_id)
        if key:
            shared.opts.set(key_option_name(provider_id), key)
            shared.opts.save()
        adapters[provider_id] = make_adapter(cfg)
        log.info(f"Cloud: provider added id={cfg.id} name={cfg.name!r} preset={preset} has_key={bool(key)}")
        return cfg


def update_provider(provider_id: str, **kwargs) -> ProviderConfig | None:
    """Update name / base_url / key / enabled. Rebuilds adapter on base_url or key change."""
    with cache_lock:
        existing = read_providers()
        cfg = next((p for p in existing if p.id == provider_id), None)
        if cfg is None:
            log.warning(f"Cloud: update_provider not found id={provider_id}")
            return None
        rebuild = False
        if "name" in kwargs:
            cfg.name = kwargs["name"]
        if "base_url" in kwargs:
            new_val = (kwargs["base_url"] or "").rstrip("/")
            if new_val != cfg.base_url:
                cfg.base_url = new_val
                rebuild = True
        if "enabled" in kwargs:
            cfg.enabled = bool(kwargs["enabled"])
        if "key" in kwargs:
            register_key_option(provider_id)
            shared.opts.set(key_option_name(provider_id), kwargs["key"] or "")
            shared.opts.save()
            rebuild = True
        write_providers(existing)
        if rebuild or not cfg.enabled:
            existing_adapter = adapters.pop(provider_id, None)
            if existing_adapter is not None:
                try:
                    existing_adapter.close()
                except Exception as e:
                    log.warning(f"Cloud: adapter close error provider={provider_id}: {e}")
        log.info(f"Cloud: provider updated id={cfg.id} fields={list(kwargs)} enabled={cfg.enabled}")
        return cfg


def remove_provider(provider_id: str) -> bool:
    with cache_lock:
        existing = read_providers()
        new_list = [p for p in existing if p.id != provider_id]
        if len(new_list) == len(existing):
            log.warning(f"Cloud: remove_provider not found id={provider_id}")
            return False
        # Purge the per-provider key from in-memory secrets BEFORE write_providers
        # triggers its threaded save. write_providers's save_atomic snapshots
        # `self.data | self.secrets` at execution time, so the on-disk secrets.json
        # ends up with the orphan removed in a single consistent write.
        key_name = key_option_name(provider_id)
        if key_name in shared.opts.secrets:
            del shared.opts.secrets[key_name]
        write_providers(new_list)
        adapter = adapters.pop(provider_id, None)
        if adapter is not None:
            try:
                adapter.close()
            except Exception as e:
                log.warning(f"Cloud: adapter close error provider={provider_id}: {e}")
        log.info(f"Cloud: provider removed id={provider_id}")
        return True


def refresh_models(provider_id: str) -> list[dict]:
    """Invalidate the adapter's model-list cache, then re-fetch."""
    adapter = get_adapter(provider_id)
    if hasattr(adapter, "transport"):
        adapter.transport.invalidate_cache("/v1/models")
    return adapter.list_models()


def validate_provider(provider_id: str) -> tuple[bool, str | None]:
    """Probe the configured key. Returns (valid, error_message)."""
    try:
        adapter = get_adapter(provider_id)
    except ValueError as e:
        return (False, str(e))
    try:
        ok = adapter.validate_key()
        return (ok, None if ok else "Validation failed (see server logs)")
    except CloudError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"Unexpected error: {e}")
