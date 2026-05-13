"""Cloud video generation orchestrator.

Drives the adapter's generate_video() and persists results to disk via the
same FilenameGenerator token expansion as the cloud image path. Writes raw
bytes directly with an atomic open/write rather than going through
modules.video_models.video_save.save_video, which expects a
StableDiffusionProcessingVideo object with local-pipeline fields that don't
apply to cloud.

Sync. The shared.state title is 'Cloud-Video'. outdir_cloud_video falls
through to outdir_video when empty. Cloud-specific FilenameGenerator tokens
[cloud_provider], [cloud_model], [cloud_video_duration], [cloud_video_aspect]
work via the apply_p fallback at modules/image/namegen.py:306-313.
"""

import os
import random
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

from modules import shared, paths
from modules.image.namegen import FilenameGenerator
from modules.logger import log

from modules.cloud import registry
from modules.cloud.errors import CloudError, ProviderError
from modules.cloud.protocol import CloudUsage, ProgressCallback


STATE_TITLE = "Cloud-Video"


@dataclass
class CloudVideoGenResult:
    """Result of a cloud video generation. Returned by generate_video()."""
    video: bytes = b""                  # raw bytes (mp4 typically)
    saved_path: str | None = None       # disk path when save_to_disk=True
    thumbnail: bytes | None = None      # PNG first-frame thumbnail
    duration: float | None = None
    format: str = "mp4"
    provider: str = ""
    model: str = ""
    seed: int = -1
    usage: CloudUsage | None = None
    info: dict = field(default_factory=dict)


def parse_size(size: str | None) -> tuple[int, int] | None:
    """Parse '1280x720' into (1280, 720). Returns None if unparseable."""
    if not size:
        return None
    match = re.match(r"^(\d+)\s*[xX]\s*(\d+)$", size.strip())
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def resolve_outdir() -> str:
    """Resolve cloud video outdir.

    Empty outdir_cloud_video falls through to local outdir_video rather than
    to outdir_save (which would be FilenameGenerator's own fallback).
    """
    cloud_specific = shared.opts.outdir_cloud_video or ""
    if not cloud_specific:
        cloud_specific = shared.opts.outdir_video
    return paths.resolve_output_path(shared.opts.outdir_samples, cloud_specific)


def make_synthetic_p(prompt: str, provider_id: str, model: str, seed: int,
                     duration: float | None, aspect_ratio: str | None,
                     size: str | None) -> SimpleNamespace:
    """Build a SimpleNamespace with the fields FilenameGenerator consumes.

    Mirrors modules.cloud.image.make_synthetic_p with three additional cloud
    tokens: [cloud_video_duration], [cloud_video_aspect], [cloud_video_size]
    via apply_p fallback. Width/height come from `size` so that local tokens
    `[width]`/`[height]` resolve when the user passes "1280x720".
    """
    width, height = parse_size(size) or (0, 0)
    p = SimpleNamespace()
    p.prompt = prompt
    p.negative_prompt = ""
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
    p.steps = 0
    p.cfg_scale = 0
    p.pag_scale = 0
    p.clip_skip = 1
    p.denoising_strength = 0
    p.sampler_name = ""
    p.styles = []
    p.extra_generation_params = {
        "Cloud Provider": provider_id,
        "Cloud Model": model,
        "Cloud Video Duration": duration,
        "Cloud Video Aspect": aspect_ratio,
        "Cloud Video Size": size,
    }
    p.job_timestamp = shared.state.job_timestamp
    p.watermark_text = ""    # video doesn't go through PIL watermark path
    p.watermark_image = ""
    # Custom filename tokens via apply_p fallback
    p.cloud_provider = provider_id
    p.cloud_model = model
    p.cloud_video_duration = duration
    p.cloud_video_aspect = aspect_ratio
    p.cloud_video_size = size
    return p


def build_video_filename(p: SimpleNamespace, outdir: str, extension: str) -> str:
    """Compute the on-disk filename via FilenameGenerator.

    Mirrors modules/image/save.py:173-191 sanitize/sequence/sanitize chain
    so cloud videos slot into the same gallery numbering as local images.
    """
    namegen = FilenameGenerator(p, p.seed, p.prompt, None, grid=False)
    pattern = shared.opts.samples_filename_pattern or "[seq]-[prompt_words]"
    decoration = namegen.apply(pattern)
    if not decoration:
        decoration = f"video-{p.seed}"
    filename = os.path.join(outdir, f"{decoration}.{extension}")
    filename = namegen.sanitize(filename)
    filename = namegen.sequence(filename)
    return namegen.sanitize(filename)


def write_video_bytes(video_bytes: bytes, target_path: str) -> str:
    """Write video bytes atomically (write to temp, fsync, rename).

    Mirrors modules/video_models/video_save.py:243-253 minus the local
    processing-object surface. Returns the final path.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = f"{target_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(video_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)
    log.info(f"Cloud: video saved path={target_path} bytes={len(video_bytes)}")
    return target_path


def extract_thumbnail(video_path: str) -> bytes | None:
    """Extract first frame of an mp4 as PNG bytes via cv2.

    cv2 is already a sdnext dependency (modules/image/util.py); no new reqs.
    Returns None on any failure - thumbnails are best-effort and a missing
    one should not fail the request.
    """
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            log.warning(f"Cloud: thumbnail extraction empty path={video_path}")
            return None
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            return None
        return buf.tobytes()
    except Exception as e:
        log.warning(f"Cloud: thumbnail extraction failed path={video_path}: {e}")
        return None


def write_thumbnail(video_path: str, thumb_bytes: bytes | None) -> str | None:
    """Persist the thumbnail next to the video as <video_path>.thumb.png."""
    if not thumb_bytes:
        return None
    thumb_path = f"{video_path}.thumb.png"
    try:
        with open(thumb_path, "wb") as f:
            f.write(thumb_bytes)
        return thumb_path
    except Exception as e:
        log.warning(f"Cloud: thumbnail write failed path={thumb_path}: {e}")
        return None


def build_infotext(prompt: str, provider_id: str, model: str, seed: int,
                   duration: float | None, aspect_ratio: str | None,
                   size: str | None, has_init_image: bool) -> str:
    """sdnext-style infotext sidecar for cloud-saved videos."""
    lines = [prompt or ""]
    fields = [
        f"Seed: {seed}",
        f"Cloud Provider: {provider_id}",
        f"Cloud Model: {model}",
    ]
    if duration is not None:
        fields.append(f"Duration: {duration}s")
    if aspect_ratio:
        fields.append(f"Aspect: {aspect_ratio}")
    if size:
        fields.append(f"Size: {size}")
    fields.append(f"Mode: {'i2v' if has_init_image else 't2v'}")
    lines.append(", ".join(fields))
    return "\n".join(lines)


def generate_video(
    prompt: str,
    provider_id: str,
    model: str,
    *,
    aspect_ratio: str | None = None,
    duration: float | None = None,
    size: str | None = None,
    init_image: bytes | None = None,
    seed: int = -1,
    extra_params: dict | None = None,
    save_to_disk: bool = True,
    on_progress: ProgressCallback | None = None,
) -> CloudVideoGenResult:
    """Generate a video via a cloud provider, optionally saving to disk.

    init_image (bytes) triggers image-to-video (i2v); without it the call is
    text-to-video (t2v). The api_v1 layer decodes base64 before invoking.

    Synchronous; blocks for the polling duration (5-15 minutes typical).
    Honours shared.state.interrupted between adapter-side polls.

    Raises modules.cloud.errors.* on provider failures or empty responses.
    """
    if not prompt or not prompt.strip():
        raise ValueError("generate_video: prompt is empty")
    is_i2v = init_image is not None
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    outdir = resolve_outdir()

    jobid = shared.state.begin(STATE_TITLE, api=True)
    shared.state.textinfo = f"Cloud: {provider_id} / {model}"
    log.info(f"Cloud: generate_video provider={provider_id} model={model} mode={'i2v' if is_i2v else 't2v'} duration={duration} aspect={aspect_ratio} size={size} save={save_to_disk}")

    try:
        adapter_params: dict = {
            "model": model,
            "prompt": prompt,
            "seed": seed,
        }
        if duration is not None:
            adapter_params["duration"] = duration
        if aspect_ratio:
            adapter_params["aspect_ratio"] = aspect_ratio
        if size:
            adapter_params["size"] = size
        if is_i2v:
            adapter_params["image"] = init_image
        if extra_params:
            adapter_params["extra_params"] = extra_params

        def progress_cb(event: dict) -> None:
            phase = event.get("phase", "")
            progress = event.get("progress")
            if progress is not None:
                shared.state.textinfo = f"Cloud: {provider_id} / {model} - {phase} ({progress * 100:.0f}%)"
            else:
                shared.state.textinfo = f"Cloud: {provider_id} / {model} - {phase}"
            if on_progress is not None:
                on_progress(event)

        adapter = registry.get_adapter(provider_id)
        result = adapter.generate_video(adapter_params, progress_cb)

        if not result.data:
            raise ProviderError(
                f"Provider returned empty video for prompt={prompt[:60]!r}",
                provider=provider_id,
            )

        saved_path: str | None = None
        thumbnail_bytes: bytes | None = None
        if save_to_disk:
            shared.state.textinfo = f"Cloud: {provider_id} / {model} - saving"
            p = make_synthetic_p(prompt, provider_id, model, seed,
                                 duration, aspect_ratio, size)
            target_path = build_video_filename(p, outdir, result.format)
            saved_path = write_video_bytes(result.data, target_path)
            # Infotext sidecar so cloud videos round-trip metadata.
            try:
                txt_path = os.path.splitext(saved_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(build_infotext(prompt, provider_id, model, seed,
                                           duration, aspect_ratio, size, is_i2v))
            except Exception as e:
                log.warning(f"Cloud: video infotext write failed path={saved_path}: {e}")
            thumbnail_bytes = extract_thumbnail(saved_path)
            write_thumbnail(saved_path, thumbnail_bytes)

        info = {
            "provider": provider_id,
            "model": model,
            "prompt": prompt,
            "seed": seed,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "size": size,
            "is_i2v": is_i2v,
            "format": result.format,
            "provider_duration": result.duration,
        }
        if result.usage is not None:
            info["usage"] = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cost": result.usage.cost,
            }

        return CloudVideoGenResult(
            video=result.data,
            saved_path=saved_path,
            thumbnail=thumbnail_bytes,
            duration=result.duration,
            format=result.format,
            provider=provider_id,
            model=model,
            seed=seed,
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
