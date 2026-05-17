"""Cloud image size constraint probe: main CLI entry point.

Usage:
    python -m modules.cloud.probe.probe_image_sizes --provider nanogpt [--dry-run | --discovery-only]
    python -m modules.cloud.probe.probe_image_sizes --provider openrouter --models flux-1.1-pro,flux-2-pro
    python -m modules.cloud.probe.probe_image_sizes --provider aihubmix --resume

Workflow:
    1. Discovery pass (free, no generation): writes test/cloud/discovery/<provider>/*.json
    2. Family grouping (deterministic): groups by last-hyphen-segment stripping
    3. Codify from metadata where unambiguous (no generation)
    4. Generation probes only for uncodified families; one cheapest variant per family
    5. Replicate constraint to siblings with 'inferred_from' attribution
    6. Write all entries to modules/cloud/size_constraints.json.in-progress; os.replace at end

This module is dev tooling, not loaded at sdnext runtime. Do not run against
a live sdnext instance: concurrent writes to size_constraints.json risk a
torn load even with atomic-replace semantics (live sdnext caches via
@functools.cache, so the race window is only on cold startup, but still).
"""

# The sdnext modules.* import chain calls cmd_args.parse_args() at shared.py
# import time, which rejects argparse args that aren't sdnext's own. Reset
# sys.argv to just the program name BEFORE importing anything from modules.*,
# then re-parse the original argv with our own parser below.
import sys

ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]

import argparse
import json
import os
import time
from datetime import datetime, timezone

from modules.cloud import registry
from modules.cloud.adapter import SIZE_CONSTRAINTS_PATH
from modules.cloud.probe import codify, grouping
from modules.cloud.probe.discovery import harvest_provider_metadata
from modules.cloud.protocol import SizeConstraint
from modules.logger import log


IN_PROGRESS_PATH = SIZE_CONSTRAINTS_PATH.with_suffix(".json.in-progress")
SCHEMA_VERSION = 1
PROBE_VARIANTS_DEFAULT = ("valid_default", "above_max", "below_min")
GENERATION_PROBE_BACKOFF_SECONDS = 30.0


def cheapest_family_member(family_members: list[dict]) -> dict | None:
    """Pick the family member with the lowest per-image price for probing.

    Pricing key precedence: per_image > per_request > first alphabetical fallback.
    Returns None for an empty family.
    """
    if not family_members:
        return None

    def cost(m: dict) -> float:
        pricing = m.get("pricing") or {}
        for key in ("per_image", "per_request"):
            val = pricing.get(key)
            try:
                return float(val) if val is not None else float("inf")
            except (TypeError, ValueError):
                continue
        return float("inf")

    sorted_by_cost = sorted(family_members, key=lambda m: (cost(m), m.get("id", "")))
    return sorted_by_cost[0]


def serialize_constraint_entry(constraint: SizeConstraint, source: str, inferred_from: str | None = None) -> dict:
    """Serialize a SizeConstraint plus probe-provenance metadata for the JSON
    entries map. `source` is one of: 'metadata', 'probed', 'inferred_from_family'."""
    payload = constraint.model_dump(exclude_none=True)
    payload["_source"] = source
    if inferred_from:
        payload["_inferred_from"] = inferred_from
    return payload


def load_in_progress() -> dict:
    """Read the .in-progress checkpoint file if it exists; otherwise the empty wrapper."""
    if IN_PROGRESS_PATH.exists():
        try:
            return json.loads(IN_PROGRESS_PATH.read_text())
        except json.JSONDecodeError as e:
            log.warning(f"Cloud probe: in-progress checkpoint malformed ({e}); starting fresh")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(),
        "entries": {},
    }


def save_in_progress(payload: dict) -> None:
    """Write the checkpoint file. Updates `generated` timestamp on each save."""
    payload["generated"] = datetime.now(timezone.utc).isoformat()
    IN_PROGRESS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def finalize_in_progress() -> None:
    """Atomic-replace size_constraints.json with the in-progress file.

    Run only after a successful end-of-sweep. The atomic rename ensures
    sdnext sees either the old or the new file in full, never a torn partial
    state, sidestepping any lock-contention question.
    """
    if not IN_PROGRESS_PATH.exists():
        log.warning("Cloud probe: finalize called but no in-progress file exists; nothing to do")
        return
    os.replace(IN_PROGRESS_PATH, SIZE_CONSTRAINTS_PATH)
    log.info(f"Cloud probe: finalized -> {SIZE_CONSTRAINTS_PATH}")


def run_generation_probe(adapter, model_id: str, variant: str) -> dict:
    """Stub for the live generation probe. Returns a dict describing the result.

    Lands as NotImplementedError; the live probe implementation replaces this
    with actual /v1/images/generations calls. Keeping it isolated here means
    the orchestrator's flow is test-coverable without network mocking.
    """
    raise NotImplementedError(
        f"Generation probe not implemented; called with model={model_id} variant={variant}. "
        "Implement when a live probe run is needed."
    )


def probe_provider(
    provider_id: str,
    *,
    only_models: list[str] | None = None,
    discovery_only: bool = False,
    dry_run: bool = False,
    resume: bool = False,
) -> dict:
    """Run the full probe workflow for one provider.

    Returns the final entries map for the provider's models (also written to
    the in-progress checkpoint). Caller is responsible for calling
    finalize_in_progress() once all providers have been processed.
    """
    if dry_run:
        log.info(f"Cloud probe: dry-run mode provider={provider_id}; no API calls will be made")

    adapter = registry.get_adapter(provider_id)
    if dry_run:
        log.info(f"Cloud probe: would call discovery for provider={provider_id}")
        return {}

    payload = load_in_progress() if resume else {"schema_version": SCHEMA_VERSION, "generated": datetime.now(timezone.utc).isoformat(), "entries": {}}
    entries: dict = payload.setdefault("entries", {})

    discovery = harvest_provider_metadata(adapter, filter_image_modality=True)
    if only_models:
        discovery = {mid: data for mid, data in discovery.items() if mid in only_models}
        if not discovery:
            log.warning(f"Cloud probe: --models filter matched no models on provider={provider_id}")
            return {}

    families = grouping.group_models_by_family(discovery.keys())
    log.info(f"Cloud probe: provider={provider_id} models={len(discovery)} families={len(families)}")

    for family_key, members in families.items():
        member_data = [discovery[mid] for mid in members]
        codified_from_metadata: dict[str, SizeConstraint] = {}
        for mid, data in zip(members, member_data):
            constraint = codify.codify_from_hints(data["hints"])
            if constraint is not None:
                codified_from_metadata[mid] = constraint
                entries[f"{provider_id}/{mid}"] = serialize_constraint_entry(constraint, source="metadata")
                log.info(f"Cloud probe: codified from metadata {provider_id}/{mid} kind={constraint.kind}")

        uncodified_members = [mid for mid in members if mid not in codified_from_metadata]
        if not uncodified_members:
            save_in_progress(payload)
            continue

        if discovery_only:
            log.info(f"Cloud probe: discovery-only mode; skipping generation probes for family={family_key} ({len(uncodified_members)} uncodified)")
            continue

        rep_normalized = cheapest_family_member([discovery[mid]["normalized"] for mid in uncodified_members])
        if rep_normalized is None:
            continue
        rep_id = rep_normalized["id"]
        log.info(f"Cloud probe: family={family_key} representative={rep_id} (cheapest of {len(uncodified_members)} uncodified)")

        try:
            probed_constraint = probe_single_model(adapter, rep_id)  # pylint: disable=assignment-from-none
        except NotImplementedError as e:
            log.warning(f"Cloud probe: {e}")
            continue
        except Exception as e:
            log.error(f"Cloud probe: probe failed family={family_key} representative={rep_id}: {e}")
            time.sleep(GENERATION_PROBE_BACKOFF_SECONDS)
            continue

        if probed_constraint is None:
            log.warning(f"Cloud probe: probe inconclusive family={family_key} representative={rep_id}; no entry written")
            continue

        entries[f"{provider_id}/{rep_id}"] = serialize_constraint_entry(probed_constraint, source="probed")
        for sibling_id in uncodified_members:
            if sibling_id == rep_id:
                continue
            entries[f"{provider_id}/{sibling_id}"] = serialize_constraint_entry(probed_constraint, source="inferred_from_family", inferred_from=rep_id)
        log.info(f"Cloud probe: replicated constraint kind={probed_constraint.kind} to {len(uncodified_members) - 1} siblings of {rep_id}")

        save_in_progress(payload)

    return entries


def probe_single_model(adapter, model_id: str) -> SizeConstraint | None:
    """Run the per-variant generation probe set for a single model.

    Defers per-variant work to run_generation_probe (currently
    NotImplementedError); a live implementation wires actual probe variants
    here.
    """
    results: dict[str, dict] = {}
    for variant in PROBE_VARIANTS_DEFAULT:
        results[variant] = run_generation_probe(adapter, model_id, variant)
    # Codify probe results into a SizeConstraint here once the live probe lands.
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="probe_image_sizes", description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--provider", required=True, help="Provider id (e.g. nanogpt, openrouter, aihubmix)")
    parser.add_argument("--models", default=None, help="Comma-separated model ID allowlist (default: all image-capable)")
    parser.add_argument("--discovery-only", action="store_true", help="Discovery pass only; no generation probes (free)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without any API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from .in-progress checkpoint if present")
    parser.add_argument("--finalize", action="store_true", help="Atomically replace size_constraints.json with .in-progress and exit")
    return parser.parse_args(argv[1:])


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = ORIGINAL_ARGV
    args = parse_args(argv)

    if args.finalize:
        finalize_in_progress()
        return 0

    only_models = [m.strip() for m in args.models.split(",")] if args.models else None
    try:
        probe_provider(
            args.provider,
            only_models=only_models,
            discovery_only=args.discovery_only,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    except Exception as e:
        log.error(f"Cloud probe: aborted: {e}")
        return 1
    if not args.dry_run and not args.discovery_only:
        log.info(f"Cloud probe: complete. Run with --finalize to replace {SIZE_CONSTRAINTS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
