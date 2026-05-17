"""Convert metadata hints into SizeConstraint instances where possible.

Pure functions, no I/O. Tested via test_probe_codify.py.

The discovery pass (discovery.py) produces a `MetadataHints` dict per model
from the provider's bulk list, per-model detail endpoint, pricing, and
description. This module turns hints into SizeConstraint variants when the
shape is unambiguous; returns None when generation probing is required to
disambiguate.

Codification rules (conservative: when in doubt, return None so the caller
runs a generation probe):

  enum     when hints carry an explicit resolutions list of WxH strings
  bucket   when hints carry symbolic-label sizing (e.g. "1k"/"2k"/"4k")
  free     not codifiable from metadata alone; always requires probing
  None     hints insufficient; generation probe required
"""

import re

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

    All strings must be uniformly WxH literals OR uniformly bucket labels.
    Mixed lists are treated as ambiguous and return None.
    """
    if not resolutions:
        return None
    cleaned = [r.strip() for r in resolutions if r and r.strip()]
    if not cleaned:
        return None
    all_wxh = all(looks_like_wxh(r) for r in cleaned)
    all_bucket = all(looks_like_bucket_label(r) for r in cleaned)
    if all_wxh:
        normalized = [normalize_wxh(r) or r for r in cleaned]
        chosen_default = default if default in normalized else None
        return SizeConstraintEnum(options=normalized, default=chosen_default)
    if all_bucket:
        normalized = [r.lower() for r in cleaned]
        resolve = {label: dims for label in normalized if (dims := bucket_label_to_dims(label)) is not None}
        chosen_default = default if default in normalized else None
        return SizeConstraintBucket(options=normalized, resolve=resolve, default=chosen_default)
    return None  # mixed shapes - probe required


def codify_from_pricing_keys(pricing_keys: list[str]) -> SizeConstraint | None:
    """Some providers (Replicate-style) charge per-resolution; the pricing dict
    keys themselves are resolution labels. If those labels look like WxH or
    bucket labels, codify directly."""
    return codify_from_resolutions(pricing_keys)


def codify_from_hints(hints: dict) -> SizeConstraint | None:
    """Top-level entry. Tries each codification signal in priority order;
    returns the first successful match or None.

    Priority:
      1. hints["resolutions"] (most provider-explicit signal)
      2. hints["sizes"] (alias some providers use)
      3. hints["pricing_keys"] (Replicate-style per-resolution charging)

    Caller propagates allow_auto / auto_wire / default from elsewhere if a
    constraint comes back; codify only handles the kind+options/resolve.
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
