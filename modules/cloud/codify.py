"""Convert provider model metadata into SizeConstraint instances.

Pure functions, no I/O. Runtime path consumed by adapter.normalize_models;
also reused by the probe-tooling discovery pass for snapshot generation.

The codifier operates on a `MetadataHints` dict extracted by
extract_hints_from_model from the raw provider model object plus the
already-extracted supported_params list. Returns a SizeConstraint variant
when the shape is unambiguous; returns None when no size data is present
or when the shape is mixed (cannot disambiguate without a probe).

Codification rules (conservative: when in doubt, return None):

  enum     when hints carry an explicit resolutions list of WxH strings
  bucket   when hints carry symbolic-label sizing (e.g. "1k"/"2k"/"4k")
  free     not codifiable from metadata alone; would require a probe
  None     hints insufficient; pre-flight short-circuits (no constraint)
"""

import re
from typing import Any, Literal

from modules.cloud.protocol import (
    SizeConstraint,
    SizeConstraintBucket,
    SizeConstraintEnum,
)


WIDTHxHEIGHT_PATTERN = re.compile(r"^(\d{2,5})\s*[xX]\s*(\d{2,5})$")
SYMBOLIC_BUCKET_PATTERN = re.compile(r"^(\d{1,2})([kK])$")


UNICODE_MULTIPLICATION_SIGN = "×"  # U+00D7 MULTIPLICATION SIGN  # noqa: RUF001


def normalize_separator(value: str) -> str:
    """Replace Unicode multiplication sign with ASCII x; some providers use it."""
    return value.strip().replace(UNICODE_MULTIPLICATION_SIGN, "x")


def looks_like_wxh(value: str) -> bool:
    """True if value matches a WxH literal like '1024x1024' or '1536X1024'."""
    return bool(WIDTHxHEIGHT_PATTERN.match(normalize_separator(value)))


def looks_like_bucket_label(value: str) -> bool:
    """True if value matches symbolic bucket sizing like '1k', '2K', '4k'."""
    return bool(SYMBOLIC_BUCKET_PATTERN.match(value.strip()))


def normalize_wxh(value: str) -> str | None:
    """Normalize a WxH-looking string to canonical '<w>x<h>' form, or None."""
    m = WIDTHxHEIGHT_PATTERN.match(normalize_separator(value))
    if not m:
        return None
    return f"{int(m.group(1))}x{int(m.group(2))}"


def bucket_label_to_dims(label: str) -> dict[str, int] | None:
    """Convert '2k' -> {'w': 2048, 'h': 2048}; returns None for non-bucket labels."""
    m = SYMBOLIC_BUCKET_PATTERN.match(label.strip())
    if not m:
        return None
    side = int(m.group(1)) * 1024
    return {"w": side, "h": side}


def codify_from_resolutions(resolutions: list[str], default: str | None = None) -> SizeConstraint | None:
    """Choose an Enum or Bucket variant from a list of resolution-like strings.

    The string 'auto' (case-insensitive) is extracted as an `allow_auto=True`
    signal regardless of where it sits in the list; providers commonly list
    it alongside concrete WxH options (e.g. NanoGPT's Qwen Image 2.0 Pro:
    ['1024x1024', '1280x720', ..., 'auto']). After removing 'auto', the
    remaining entries must be uniformly WxH literals OR uniformly bucket
    labels; mixed remainders return None.

    An auto-only list (just ['auto']) yields an Enum with empty options and
    allow_auto=True.
    """
    if not resolutions:
        return None
    cleaned = [r.strip() for r in resolutions if r and r.strip()]
    if not cleaned:
        return None
    allow_auto = any(r.lower() == "auto" for r in cleaned)
    non_auto = [r for r in cleaned if r.lower() != "auto"]
    auto_wire: Literal["literal", "omit", "default"] | None = "literal" if allow_auto else None

    if not non_auto:
        if default and default.lower() == "auto":
            default = None
        return SizeConstraintEnum(options=[], allow_auto=True, auto_wire=auto_wire, default=default)

    all_wxh = all(looks_like_wxh(r) for r in non_auto)
    all_bucket = all(looks_like_bucket_label(r) for r in non_auto)
    if all_wxh:
        normalized = [normalize_wxh(r) or r for r in non_auto]
        chosen_default = default if default in normalized else None
        return SizeConstraintEnum(options=normalized, allow_auto=allow_auto, auto_wire=auto_wire, default=chosen_default)
    if all_bucket:
        normalized = [r.lower() for r in non_auto]
        resolve = {label: dims for label in normalized if (dims := bucket_label_to_dims(label)) is not None}
        chosen_default = default if default in normalized else None
        return SizeConstraintBucket(options=normalized, resolve=resolve, allow_auto=allow_auto, auto_wire=auto_wire, default=chosen_default)
    return None  # mixed remainder; constraint unrecoverable from metadata alone


def codify_from_pricing_keys(pricing_keys: list[str]) -> SizeConstraint | None:
    """Some providers (Replicate-style) charge per-resolution; the pricing dict
    keys themselves are resolution labels. If those labels look like WxH or
    bucket labels, codify directly."""
    return codify_from_resolutions(pricing_keys)


def codify_from_hints(hints: dict) -> SizeConstraint | None:
    """Top-level entry. Tries each codification signal in priority order;
    returns the first successful match or None.

    Priority:
      1. hints["resolutions"]    (most provider-explicit signal)
      2. hints["sizes"]          (alias some providers use)
      3. hints["size_options"]   (alias some providers use)
      4. hints["pricing_keys"]   (Replicate-style per-resolution charging)
    """
    for key in ("resolutions", "sizes", "size_options"):
        if isinstance(hints.get(key), list):
            constraint = codify_from_resolutions(hints[key], default=hints.get("default_size"))
            if constraint is not None:
                return constraint
    pricing_keys = hints.get("pricing_keys")
    if isinstance(pricing_keys, list):
        constraint = codify_from_pricing_keys(pricing_keys)
        if constraint is not None:
            return constraint
    return None


def extract_hints_from_model(raw_model: dict, supported_params: list[dict] | None) -> dict[str, Any]:
    """Build a hints dict from a raw provider model object plus already-
    extracted supported_params list.

    Sources, in priority order:
      - supported_params entry where name=='size' and type=='enum'
      - top-level raw_model['supported_parameters']['resolutions'] (NanoGPT)
      - raw_model['pricing']['per_image'].keys() (NanoGPT/Replicate-style)
      - raw_model['description'] (last-resort hint for human inspection)
    """
    hints: dict[str, Any] = {}
    supported_params = supported_params or []
    for param in supported_params:
        if param.get("name") == "size" and param.get("type") == "enum":
            options = param.get("options") or []
            if options:
                hints["resolutions"] = list(options)
            if param.get("default"):
                hints["default_size"] = param["default"]
            break
    if "resolutions" not in hints:
        raw_supported = raw_model.get("supported_parameters")
        if isinstance(raw_supported, dict):
            res = raw_supported.get("resolutions")
            if isinstance(res, list) and res:
                hints["resolutions"] = list(res)
    pricing = raw_model.get("pricing")
    if isinstance(pricing, dict):
        per_image = pricing.get("per_image")
        if isinstance(per_image, dict):
            res_like_keys = [k for k in per_image.keys() if isinstance(k, str) and looks_like_wxh(k.replace("*", "x"))]
            if res_like_keys:
                hints["pricing_keys"] = [k.replace("*", "x") for k in res_like_keys]
    description = raw_model.get("description") or raw_model.get("desc")
    if description:
        hints["description"] = description
    return hints


def codify_from_model(raw_model: dict, supported_params: list[dict] | None) -> SizeConstraint | None:
    """One-shot entry for adapter.normalize_models: extract hints + codify."""
    return codify_from_hints(extract_hints_from_model(raw_model, supported_params))
