"""Family detection for cloud image models.

Pure functions, no I/O. Probed via test_probe_grouping.py.

A "family" is a set of model IDs sharing a stem with only a final variant
suffix differing. The rule: strip the last hyphen-segment to obtain the
family key, falling back to the full ID when the last segment can't
reasonably be considered a variant marker.

Examples:
  flux-1.1-pro          -> family flux-1.1
  flux-1.1-pro-ultra    -> family flux-1.1-pro
  flux-1.1-dev          -> family flux-1.1
  seedream-4-5          -> family seedream-4    (acceptable; one-member family is fine)
  nano-banana           -> family nano
  pruna-ai/flux-1.1-pro -> family pruna-ai/flux-1.1
"""

from collections.abc import Iterable

# A trailing segment is treated as a *variant* marker (dev/pro/ultra/turbo)
# and stripped to obtain the family key only if it is pure alphabetic and at
# least this many characters. The pure-letters check protects version markers
# like "v3" or "5" from being mistaken for variant suffixes: `dall-e-3` and
# `dall-e-2` stay as their own families, while `flux-1.1-pro` and
# `flux-1.1-dev` correctly group under `flux-1.1`.
MIN_VARIANT_SEGMENT_LENGTH = 2


def family_of(model_id: str) -> str:
    """Return the family key for a model ID by stripping the last hyphen-segment.

    Stripping rule: the trailing segment must be pure alphabetic (no digits,
    no `v3`-style version markers) and at least MIN_VARIANT_SEGMENT_LENGTH
    characters long. The model ID's path portion (after the last slash) is
    what gets segmented; any namespace prefix (e.g. "pruna-ai/") is preserved
    verbatim so families in different providers' namespaces never collide.
    """
    if "/" in model_id:
        prefix, _, leaf = model_id.rpartition("/")
        return f"{prefix}/{family_of(leaf)}"
    if "-" not in model_id:
        return model_id
    stem, _, last = model_id.rpartition("-")
    if not stem or not last.isalpha() or len(last) < MIN_VARIANT_SEGMENT_LENGTH:
        return model_id
    return stem


def group_models_by_family(model_ids: Iterable[str]) -> dict[str, list[str]]:
    """Group model IDs into families. Preserves input order within each family."""
    groups: dict[str, list[str]] = {}
    for mid in model_ids:
        key = family_of(mid)
        groups.setdefault(key, []).append(mid)
    return groups


def is_family_singleton(group: list[str]) -> bool:
    """A one-member family is treated identically to a multi-member one for
    constraint propagation, but callers may want to log differently."""
    return len(group) == 1
