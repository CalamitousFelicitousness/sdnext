"""Cloud image generation orchestrator.

Drives the adapter's generate_image() and integrates results with sdnext's
existing image-saving pipeline (FilenameGenerator + images.save_image). The
adapter returns raw bytes; this module builds a synthetic processing object
so FilenameGenerator tokens resolve, resolves the outdir, writes to disk,
and returns a CloudImageGenResult.

Sync to match the rest of the cloud module. The shared.state title is
"Cloud-Image".
"""

import io
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace

from PIL import Image

from modules import shared, images, paths
from modules.logger import log

from modules.cloud import registry
from modules.cloud.errors import CloudError, InputValidationError, ProviderError
from modules.cloud.protocol import (
    CloudUsage,
    ProgressCallback,
    SizeConstraint,
    SizeConstraintBucket,
    SizeConstraintEnum,
    SizeConstraintFree,
)


STATE_TITLE = "Cloud-Image"

TELEMETRY_WINDOW_DAYS = 90
TELEMETRY_EVENTS = ("predicted_valid", "predicted_invalid_server_will_judge", "predicted_invalid_hard_block")


@dataclass
class CloudImageGenResult:
    """Result of a cloud image generation. Returned by generate_image()."""
    images: list[bytes] = field(default_factory=list)       # raw image bytes (PNG/JPEG/WEBP per provider)
    saved_paths: list[str] = field(default_factory=list)    # disk paths if save_to_disk=True
    revised_prompt: str | None = None
    provider: str = ""
    model: str = ""
    seed: int = -1
    width: int = 0
    height: int = 0
    usage: CloudUsage | None = None
    info: dict = field(default_factory=dict)                # serialisable metadata dict


def check_size_against_constraint(width: int, height: int, ask_auto: bool, constraint: SizeConstraint) -> tuple[bool, str]:
    """Pure validation: does (width, height) satisfy the given constraint?

    Returns (ok, reason). Reason is empty when ok=True; populated with a
    one-sentence human-readable description otherwise.
    """
    if ask_auto:
        if constraint.allow_auto:
            return True, ""
        return False, "model does not support size=auto"

    if isinstance(constraint, SizeConstraintEnum):
        size_str = f"{width}x{height}"
        if size_str in constraint.options:
            return True, ""
        return False, f"size {size_str} not in enum options {constraint.options}"

    if isinstance(constraint, SizeConstraintBucket):
        size_str = f"{width}x{height}"
        if size_str in constraint.options:
            return True, ""
        for dims in constraint.resolve.values():
            if dims.get("w") == width and dims.get("h") == height:
                return True, ""
        return False, f"size {size_str} matches neither symbolic bucket {constraint.options} nor any resolved bucket dim"

    if isinstance(constraint, SizeConstraintFree):
        pixel_count = width * height
        if constraint.min_pixel_count is not None and pixel_count < constraint.min_pixel_count:
            return False, f"pixel count {pixel_count} below min {constraint.min_pixel_count}"
        if constraint.max_pixel_count is not None and pixel_count > constraint.max_pixel_count:
            return False, f"pixel count {pixel_count} above max {constraint.max_pixel_count}"
        longest = max(width, height)
        if constraint.min_longest_side is not None and longest < constraint.min_longest_side:
            return False, f"longest side {longest} below min {constraint.min_longest_side}"
        if constraint.max_longest_side is not None and longest > constraint.max_longest_side:
            return False, f"longest side {longest} above max {constraint.max_longest_side}"
        if height > 0:
            ratio = width / height
            if constraint.aspect_ratio_min is not None and ratio < constraint.aspect_ratio_min:
                return False, f"aspect ratio {ratio:.3f} below min {constraint.aspect_ratio_min}"
            if constraint.aspect_ratio_max is not None and ratio > constraint.aspect_ratio_max:
                return False, f"aspect ratio {ratio:.3f} above max {constraint.aspect_ratio_max}"
        if constraint.align is not None:
            w_align, h_align = constraint.align if isinstance(constraint.align, tuple) else (constraint.align, constraint.align)
            if width % w_align != 0:
                return False, f"width {width} not aligned to {w_align}"
            if height % h_align != 0:
                return False, f"height {height} not aligned to {h_align}"
        return True, ""

    return True, ""  # unknown variant; defensive pass-through (typed union should make this unreachable)


def record_size_validation_telemetry(provider_id: str, event: str) -> None:
    """Increment a per-provider counter on the cloud_size_validation_telemetry hidden opt.

    Resets the per-provider window if older than TELEMETRY_WINDOW_DAYS. The
    opts.save() cadence (debounced) handles persistence; no separate JSON file
    or fasteners locking. Silent on any failure since telemetry is best-effort.
    """
    if not hasattr(shared, "opts"):
        return
    try:
        raw = getattr(shared.opts, "cloud_size_validation_telemetry", "{}")
        data = json.loads(raw) if isinstance(raw, str) and raw else {}
    except (json.JSONDecodeError, TypeError):
        data = {}
    bucket = data.setdefault(provider_id, {})
    now = datetime.now(timezone.utc)
    window_start_iso = bucket.get("window_start_iso")
    reset_needed = True
    if window_start_iso:
        try:
            window_start = datetime.fromisoformat(window_start_iso)
            if (now - window_start).days < TELEMETRY_WINDOW_DAYS:
                reset_needed = False
        except ValueError:
            pass
    if reset_needed:
        bucket["window_start_iso"] = now.isoformat()
        for key in TELEMETRY_EVENTS:
            bucket[key] = 0
        bucket["total"] = 0
    bucket[event] = bucket.get(event, 0) + 1
    bucket["total"] = bucket.get("total", 0) + 1
    try:
        shared.opts.cloud_size_validation_telemetry = json.dumps(data)
    except Exception as e:
        log.debug(f"Cloud: telemetry persist failed: {e}")


def resolve_default_to_dims(constraint: SizeConstraint) -> tuple[int, int]:
    """Parse a SizeConstraint's `default` value to a (width, height) pair.

    Enum and Free defaults are WxH strings ('1024x1024'). Bucket defaults are
    symbolic labels ('2k') resolved through the constraint's `resolve` map.
    Returns (0, 0) on any parse failure so callers can degrade gracefully.
    """
    default = constraint.default
    if not default:
        return 0, 0
    if isinstance(constraint, SizeConstraintBucket):
        dims = constraint.resolve.get(default) or {}
        return int(dims.get("w", 0)), int(dims.get("h", 0))
    try:
        w_str, h_str = default.lower().split("x", 1)
        return int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return 0, 0


def resolve_auto_dispatch(provider_id: str, model: str) -> tuple[str | None, int, int, str]:
    """Resolve how `size="auto"` should be wired into adapter params.

    Looks up the model's size_constraint and consults its `auto_wire` field to
    decide what shape the outgoing request body should take. Returns
    `(size_token, dispatch_width, dispatch_height, mode)`:

    - `size_token` -> value for adapter_params["size"], or None to omit the key
    - `dispatch_width`/`dispatch_height` -> values for adapter_params width/height
      (0 short-circuits the preset's `image_size_transform` so a literal "auto"
      survives into the outgoing JSON body)
    - `mode` -> one of `"literal"`, `"omit"`, `"default-resolved"`, `"unknown"`,
      used by the caller to decide whether preflight should still validate WxH

    When the model has no constraint, or has `allow_auto=False`, we still pass
    `"auto"` through as a literal: the adapter has no better signal and the
    provider's own error envelope becomes the source of truth. The pre-flight
    helper (with `ask_auto=True`) will warn or hard-block based on `allow_auto`.
    """
    from modules.cloud.adapter import get_size_constraint  # lazy: avoid import cycle
    constraint = get_size_constraint(provider_id, model)
    if constraint is None or not constraint.allow_auto:
        return "auto", 0, 0, "unknown"
    wire = constraint.auto_wire or "literal"
    if wire == "omit":
        return None, 0, 0, "omit"
    if wire == "default":
        w, h = resolve_default_to_dims(constraint)
        if w and h:
            return None, w, h, "default-resolved"
        return "auto", 0, 0, "literal"  # default missing: degrade to literal
    return "auto", 0, 0, "literal"


def run_size_preflight(provider_id: str, model: str, width: int, height: int, ask_auto: bool = False) -> None:
    """Run pre-flight size validation per the cloud_image_size_validation setting.

    The rollout is soft-then-hard: 'soft' logs a warning on mismatch but still
    dispatches the request; 'hard' raises InputValidationError before any HTTP
    call; 'off' skips validation entirely. Constraint absence (no entry for
    this provider+model in size_constraints.json) is treated as 'no opinion'
    and short-circuits.
    """
    mode = getattr(shared.opts, "cloud_image_size_validation", "soft") if hasattr(shared, "opts") else "soft"
    if mode == "off":
        return
    from modules.cloud.adapter import get_size_constraint  # lazy to avoid import-cycle
    constraint = get_size_constraint(provider_id, model)
    if constraint is None:
        return
    ok, reason = check_size_against_constraint(width, height, ask_auto, constraint)
    if ok:
        record_size_validation_telemetry(provider_id, "predicted_valid")
        return
    if mode == "hard":
        record_size_validation_telemetry(provider_id, "predicted_invalid_hard_block")
        raise InputValidationError(
            f"Cloud: requested size {width}x{height} does not satisfy {model} constraint ({reason}). Set cloud_image_size_validation=off to suppress this check.",
            provider=provider_id,
            field="size",
            limit=constraint.model_dump(exclude_none=True),
        )
    # soft mode: warn but continue
    record_size_validation_telemetry(provider_id, "predicted_invalid_server_will_judge")
    log.warning(
        f"Cloud: size_constraint mismatch provider={provider_id} model={model} requested={width}x{height} reason={reason} "
        f"(set cloud_image_size_validation=hard to enforce client-side; cloud_image_size_validation=off to suppress this warning)"
    )


def resolve_outdir(is_img2img: bool) -> str:
    """Resolve cloud image outdir.

    Empty `outdir_cloud_image` falls through to local txt2img / img2img dir
    rather than to outdir_save (which is images.save_image's own fallback).
    """
    cloud_specific = shared.opts.outdir_cloud_image or ""
    if not cloud_specific:
        cloud_specific = shared.opts.outdir_img2img_samples if is_img2img else shared.opts.outdir_txt2img_samples
    return paths.resolve_output_path(shared.opts.outdir_samples, cloud_specific)


def build_infotext(prompt: str, negative_prompt: str, provider_id: str, model: str,
                   seed: int, width: int, height: int, steps: int, guidance_scale: float,
                   revised_prompt: str | None) -> str:
    """Build the PNG `parameters` text chunk in sdnext's familiar format.

    Mirrors the local generation infotext shape so "send to txt2img" round-trips
    work for cloud images. Cloud-specific fields slot into the comma-separated
    second line.
    """
    lines = [prompt or ""]
    if negative_prompt:
        lines.append(f"Negative prompt: {negative_prompt}")
    fields = [
        f"Steps: {steps}",
        f"CFG scale: {guidance_scale}",
        f"Seed: {seed}",
        f"Size: {width}x{height}",
        f"Cloud Provider: {provider_id}",
        f"Cloud Model: {model}",
    ]
    if revised_prompt:
        fields.append(f"Revised prompt: {revised_prompt!r}")
    lines.append(", ".join(fields))
    return "\n".join(lines)


def make_synthetic_p(prompt: str, negative_prompt: str, provider_id: str, model: str,
                     seed: int, width: int, height: int, steps: int, cfg_scale: float,
                     denoising_strength: float) -> SimpleNamespace:
    """Build a SimpleNamespace with the fields FilenameGenerator + images.save_image consume.

    Do NOT subclass StableDiffusionProcessing; its init() mutates the local
    pipeline. SimpleNamespace gives us all the duck-typed attribute access we
    need with zero side effects.

    Cloud-specific FilenameGenerator tokens [cloud_provider] and [cloud_model]
    work via apply_p fallback (modules/image/namegen.py:306-313).
    """
    p = SimpleNamespace()
    # Inputs FilenameGenerator reads (modules/image/namegen.py:27-66)
    p.prompt = prompt
    p.negative_prompt = negative_prompt
    p.seed = seed
    p.all_seeds = [seed]
    p.seeds = [seed]
    p.subseed = -1
    p.all_subseeds = [-1]
    p.width = width
    p.height = height
    p.batch_size = 1
    p.n_iter = 1
    p.iteration = 0
    p.batch_index = 0
    p.steps = steps
    p.cfg_scale = cfg_scale
    p.pag_scale = 0
    p.clip_skip = 1
    p.denoising_strength = denoising_strength
    p.sampler_name = ""
    p.styles = []
    p.extra_generation_params = {
        "Cloud Provider": provider_id,
        "Cloud Model": model,
    }
    p.job_timestamp = shared.state.job_timestamp
    # Direct reads in images.save_image (image/save.py:203-205)
    p.watermark_text = shared.opts.image_watermark
    p.watermark_image = shared.opts.image_watermark_image
    # Custom tokens via apply_p fallback - enables [cloud_provider] / [cloud_model]
    # in samples_filename_pattern with no FilenameGenerator changes needed.
    p.cloud_provider = provider_id
    p.cloud_model = model
    return p


def generate_image(
    prompt: str,
    provider_id: str,
    model: str,
    *,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    ask_auto: bool = False,
    n: int = 1,
    seed: int = -1,
    steps: int = 28,
    guidance_scale: float = 7.5,
    quality: str = "standard",
    style: str | None = None,
    init_image: bytes | None = None,
    mask: bytes | None = None,
    strength: float = 0.75,
    extra_params: dict | None = None,
    save_to_disk: bool = True,
    on_progress: ProgressCallback | None = None,
) -> CloudImageGenResult:
    """Generate one or more images via a cloud provider, optionally saving to disk.

    Image and mask inputs must be raw bytes (PNG / JPEG / WEBP). The api_v1
    layer decodes from base64 before invoking. Returns a CloudImageGenResult
    with both `images` (raw bytes) and `saved_paths` (disk paths if saved).

    Raises modules.cloud.errors.* on provider failures or empty responses.
    """
    if not prompt or not prompt.strip():
        raise ValueError("generate_image: prompt is empty")
    is_img2img = init_image is not None
    if mask is not None and init_image is None:
        raise ValueError("generate_image: mask supplied without init_image")
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    outdir = resolve_outdir(is_img2img)

    # Auto-size dispatch: caller asks for provider-chosen dims;
    # consult the model's size_constraint to wire it correctly. For
    # "default-resolved" the constraint pinned a real WxH so the rest of the
    # function treats it as a normal explicit request.
    dispatch_width, dispatch_height = width, height
    dispatch_size_token: str | None = None
    dispatch_mode = "explicit"
    if ask_auto:
        dispatch_size_token, dispatch_width, dispatch_height, dispatch_mode = resolve_auto_dispatch(provider_id, model)
        if dispatch_mode == "default-resolved":
            width, height = dispatch_width, dispatch_height

    jobid = shared.state.begin(STATE_TITLE, api=True)
    shared.state.textinfo = f"Cloud: {provider_id} / {model}"
    log.info(
        f"Cloud: generate_image provider={provider_id} model={model} "
        f"{width}x{height}{' auto=' + dispatch_mode if ask_auto else ''} "
        f"n={n} mode={'img2img' if is_img2img else 'txt2img'} save={save_to_disk}"
    )

    try:
        adapter_params: dict = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": dispatch_width,
            "height": dispatch_height,
            "n": n,
            "seed": seed,
            "steps": steps,
            "guidance": guidance_scale,
            "quality": quality,
        }
        if dispatch_size_token is not None:
            adapter_params["size"] = dispatch_size_token
        if style:
            adapter_params["style"] = style
        if is_img2img:
            adapter_params["image"] = init_image
            adapter_params["strength"] = strength
            if mask is not None:
                adapter_params["mask"] = mask
        if extra_params:
            adapter_params["extra_params"] = extra_params

        def progress_cb(event: dict) -> None:
            phase = event.get("phase", "")
            shared.state.textinfo = f"Cloud: {provider_id} / {model} - {phase}"
            if on_progress is not None:
                on_progress(event)

        # In "default-resolved" mode we know the dims explicitly so validate
        # them as normal WxH; in literal/omit modes we want allow_auto checked.
        preflight_ask_auto = dispatch_mode in ("literal", "omit", "unknown")
        run_size_preflight(provider_id, model, width, height, ask_auto=preflight_ask_auto)
        adapter = registry.get_adapter(provider_id)
        result = adapter.generate_image(adapter_params, progress_cb)

        if not result.images:
            raise ProviderError(
                f"Provider returned no images for prompt={prompt[:60]!r}",
                provider=provider_id,
            )

        saved_paths: list[str] = []
        # In literal/omit auto modes the request width/height are sentinels (0)
        # and the provider chose actual dims. Track the first decoded image's
        # size so the returned result + infotext reflect reality, not sentinels.
        report_unresolved_auto = ask_auto and dispatch_mode in ("literal", "omit", "unknown")
        final_width, final_height = width, height
        first_decoded_size: tuple[int, int] | None = None
        if save_to_disk:
            shared.state.textinfo = f"Cloud: {provider_id} / {model} - saving"
            for idx, img_bytes in enumerate(result.images):
                try:
                    pil = Image.open(io.BytesIO(img_bytes))
                    pil.load()  # force decode now to surface corrupt-image errors here
                except Exception as e:
                    log.warning(f"Cloud: failed to decode image {idx+1}/{len(result.images)}: {e}")
                    continue
                if first_decoded_size is None:
                    first_decoded_size = pil.size
                display_w, display_h = (pil.size if report_unresolved_auto else (width, height))
                p = make_synthetic_p(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    provider_id=provider_id,
                    model=model,
                    seed=seed + idx,
                    width=display_w,
                    height=display_h,
                    steps=steps,
                    cfg_scale=guidance_scale,
                    denoising_strength=strength if is_img2img else 0,
                )
                p.batch_index = idx
                infotext = build_infotext(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    provider_id=provider_id,
                    model=model,
                    seed=seed + idx,
                    width=display_w,
                    height=display_h,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    revised_prompt=result.revised_prompt,
                )
                fn, _, _ = images.save_image(
                    image=pil,
                    path=outdir,
                    basename="",
                    seed=seed + idx,
                    prompt=prompt,
                    info=infotext,
                    p=p,
                )
                if fn:
                    saved_paths.append(fn)
                    log.debug(f"Cloud: saved image {idx+1}/{len(result.images)} to {fn}")
        if report_unresolved_auto and first_decoded_size is not None:
            final_width, final_height = first_decoded_size
        elif report_unresolved_auto:
            # No decode happened (e.g. save_to_disk=False). Peek the first image
            # just for dim reporting so the caller doesn't see 0x0.
            try:
                with Image.open(io.BytesIO(result.images[0])) as probe_pil:
                    final_width, final_height = probe_pil.size
            except Exception as e:
                log.debug(f"Cloud: ask_auto dim-probe failed: {e}")

        info = {
            "provider": provider_id,
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": final_width,
            "height": final_height,
            "n": n,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "quality": quality,
            "style": style,
            "is_img2img": is_img2img,
            "revised_prompt": result.revised_prompt,
            "ask_auto": ask_auto,
            "auto_dispatch": dispatch_mode if ask_auto else None,
        }
        if result.usage is not None:
            info["usage"] = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cost": result.usage.cost,
            }

        return CloudImageGenResult(
            images=result.images,
            saved_paths=saved_paths,
            revised_prompt=result.revised_prompt,
            provider=provider_id,
            model=model,
            seed=seed,
            width=final_width,
            height=final_height,
            usage=result.usage,
            info=info,
        )

    except CloudError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise ProviderError(str(e), provider=provider_id) from e
    finally:
        shared.state.end(jobid)
