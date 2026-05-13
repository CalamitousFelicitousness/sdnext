"""One-shot migration of legacy enso/cloud-providers.local.json to opts.

Triggered via script_callbacks.on_app_started (see __init__.py). Runs once
on first boot post-cutover; subsequent boots find the .migrated rename and
no-op. The callback signature is (_blocks, _app).
"""

import json
import os

from modules import shared
from modules.logger import log
from modules.paths import extensions_dir

from modules.cloud import registry


def legacy_path() -> str:
    return os.path.join(extensions_dir, "enso", "cloud-providers.local.json")


def migrate_legacy_providers(_blocks, _app) -> None:
    src = legacy_path()
    if not os.path.isfile(src):
        return
    try:
        with open(src, encoding="utf-8") as f:
            legacy = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"Cloud: legacy provider file unreadable, skipping migration src={src}: {e}")
        return

    legacy_providers = legacy.get("providers", []) or []
    if not legacy_providers:
        try:
            os.rename(src, f"{src}.migrated")
        except OSError as e:
            log.warning(f"Cloud: failed to rename empty legacy file: {e}")
        return

    migrated = 0
    skipped = 0
    with registry.cache_lock:
        current = registry.read_providers()
        existing_ids = {p.id for p in current}
        for p in legacy_providers:
            try:
                pid = p["id"]
                name = p["name"]
                preset = p["preset"]
                base_url = p["base_url"]
            except KeyError as e:
                log.warning(f"Cloud: legacy provider missing field {e}, skipping")
                continue
            if pid in existing_ids:
                skipped += 1
                continue
            current.append(registry.ProviderConfig(
                id=pid,
                name=name,
                preset=preset,
                base_url=base_url.rstrip("/"),
                enabled=p.get("enabled", True),
            ))
            existing_ids.add(pid)
            registry.register_key_option(pid)
            if p.get("key"):
                shared.opts.set(registry.key_option_name(pid), p["key"])
            migrated += 1
        if migrated:
            registry.write_providers(current)

    try:
        os.rename(src, f"{src}.migrated")
        log.info(f"Cloud: migrated {migrated} providers (skipped {skipped} already-existing) from {src}")
    except OSError as e:
        log.warning(f"Cloud: migration succeeded ({migrated} providers) but rename failed: {e}")
