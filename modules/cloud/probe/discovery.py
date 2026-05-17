"""Discovery pass: harvest model metadata without running generations.

Free or near-free per provider; hits /v1/models and per-model detail
endpoints only. No generation calls. Output is a per-model JSON file under
`test/cloud/discovery/<provider>/<sanitized_model>.json` for human review and
codification.

Two functions live here:

  harvest_provider_metadata: iterate provider model list + per-model detail,
                             produce {model_id: MetadataHints} dict.

  hints_from_normalized_model: pure helper that extracts size hints from a
                               normalized model dict produced by adapter.
                               normalize_models. Unit-testable without network.
"""

import json
from pathlib import Path
from typing import Any

from modules.logger import log


# Discovery output lives next to cassettes; git-excluded like test/cloud/.
DISCOVERY_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "test" / "cloud" / "discovery"


def sanitize_model_id_for_path(model_id: str) -> str:
    """Replace path-incompatible characters in a model ID for filesystem use.

    `/` becomes `__` so namespaced IDs like 'pruna-ai/flux-1.1-pro' become
    'pruna-ai__flux-1.1-pro' on disk.
    """
    return model_id.replace("/", "__")


def discovery_path(provider_id: str, model_id: str) -> Path:
    """Per-model discovery output path."""
    return DISCOVERY_ROOT / provider_id / f"{sanitize_model_id_for_path(model_id)}.json"


def write_discovery_artifact(provider_id: str, model_id: str, payload: dict) -> Path:
    """Save discovery output for one model. Returns the written path."""
    p = discovery_path(provider_id, model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return p


def hints_from_normalized_model(normalized: dict) -> dict[str, Any]:
    """Extract size-relevant hints from one normalize_models output entry.

    Pure function; no network. Looks at supported_params (already populated
    by extract_supported_params), pricing (already populated by
    extract_pricing), and description. Returns a dict in the shape codify.py
    expects.
    """
    hints: dict[str, Any] = {}
    supported = normalized.get("supported_params") or []
    for param in supported:
        if param.get("name") == "size" and param.get("type") == "enum":
            options = param.get("options") or []
            if options:
                hints["resolutions"] = list(options)
            if param.get("default"):
                hints["default_size"] = param["default"]
            break
    pricing = normalized.get("pricing") or {}
    pricing_keys_with_size_shape = [k for k in pricing.keys() if isinstance(k, str) and ("x" in k.lower() or k.lower().endswith("k"))]
    if pricing_keys_with_size_shape:
        hints["pricing_keys"] = pricing_keys_with_size_shape
    description = normalized.get("description") or ""
    if description:
        hints["description"] = description
    return hints


def fetch_model_detail(adapter, model_id: str) -> dict | None:
    """Try to fetch /v1/models/{id} via the adapter's transport. Returns the
    parsed JSON on 200, None on absence or error. Free per request."""
    try:
        data = adapter.transport.get_cached(f"/v1/models/{model_id}", ttl=300)
        return data if isinstance(data, dict) else None
    except Exception as e:
        log.debug(f"Cloud probe: detail fetch failed provider={adapter.provider_id} model={model_id}: {e}")
        return None


def list_aihubmix_image_models(adapter) -> list[dict]:
    """AIHubMix has no architecture metadata on /v1/models, but exposes a
    separate /api/v1/models?type=image_generation endpoint with richer per-
    model metadata (per https://docs.aihubmix.com/en/api/Models-API). Fetch
    those and normalize to the shape adapter.normalize_models would produce
    so the downstream pipeline is uniform.

    Field translation:
      model_id          -> id
      input_modalities  -> modalities (str like 'text,image' becomes list)
      types             -> modalities (image_generation forces text-to-image)
      pricing.input     -> pricing.prompt_token
      pricing.output    -> pricing.completion_token
    """
    try:
        data = adapter.transport.get_cached("/api/v1/models", ttl=300, params={"type": "image_generation"})
    except Exception as e:
        log.warning(f"Cloud probe: AIHubMix image-models endpoint failed: {e}")
        return []
    raw = data.get("data", []) if isinstance(data, dict) else []
    normalized: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        model_id = m.get("model_id") or m.get("id")
        if not model_id:
            continue
        input_mods_raw = m.get("input_modalities", "")
        input_mods = [s.strip() for s in input_mods_raw.split(",") if s.strip()] if isinstance(input_mods_raw, str) else (input_mods_raw or [])
        modalities = ["text-to-image"]
        if "image" in input_mods:
            modalities.append("image-to-image")
        pricing_raw = m.get("pricing") or {}
        pricing: dict = {"currency": "USD"}
        if pricing_raw.get("input") is not None:
            pricing["prompt_token"] = pricing_raw["input"]
        if pricing_raw.get("output") is not None:
            pricing["completion_token"] = pricing_raw["output"]
        normalized.append({
            "source": "cloud",
            "id": model_id,
            "name": m.get("model_name") or model_id,
            "provider": adapter.provider_id,
            "modalities": modalities,
            "capabilities": m.get("features") or [],
            "pricing": pricing if len(pricing) > 1 else None,
            "context_length": m.get("context_length"),
            "supported_params": None,
            "description": m.get("desc") or m.get("description"),
            "default_params": None,
            "_aihubmix_raw": m,
        })
    return normalized


def harvest_provider_metadata(adapter, *, filter_image_modality: bool = True) -> dict[str, dict]:
    """Discover metadata for every model on the given adapter.

    Returns {model_id: {"normalized": <dict>, "detail": <dict|None>, "hints": <dict>}}.

    Per-model JSON artifact also written to DISCOVERY_ROOT for human review.
    No generation calls occur. Detail-endpoint failures are logged at debug
    level and the model still gets an entry (hints derived from the bulk list
    alone).
    """
    # adapter.list_models() already returns normalized output; do NOT re-normalize.
    # Double-normalize would strip the source `architecture` field that infer_modalities
    # depends on, resulting in every model being classified as 'chat' regardless of true modality.
    # AIHubMix needs a special path because its /v1/models has no architecture metadata;
    # the dedicated /api/v1/models?type=image_generation endpoint carries proper image data.
    if adapter.provider_id == "aihubmix":
        normalized_models = list_aihubmix_image_models(adapter)
    else:
        normalized_models = adapter.list_models()
    out: dict[str, dict] = {}
    for normalized in normalized_models:
        model_id = normalized.get("id")
        if not model_id:
            continue
        if filter_image_modality:
            modalities = normalized.get("modalities") or []
            if "text-to-image" not in modalities and "image-to-image" not in modalities:
                continue
        detail = fetch_model_detail(adapter, model_id)
        hints = hints_from_normalized_model(normalized)
        if detail:
            # detail endpoint sometimes carries richer resolutions / sizes
            for key in ("resolutions", "sizes", "size_options"):
                if isinstance(detail.get(key), list) and key not in hints:
                    hints[key] = detail[key]
            if isinstance(detail.get("default_size"), str) and "default_size" not in hints:
                hints["default_size"] = detail["default_size"]
        artifact = {"model_id": model_id, "normalized": normalized, "detail": detail, "hints": hints}
        write_discovery_artifact(adapter.provider_id, model_id, artifact)
        out[model_id] = artifact
    log.info(f"Cloud probe: discovery complete provider={adapter.provider_id} models={len(out)}")
    return out
