"""OpenAI-compatible cloud provider adapter (sync).

Handles all providers that implement the OpenAI API spec (OpenRouter, OpenAI,
NanoGPT, AIHubMix, Ollama, custom). Parameterised by preset rather than
subclassed per provider.

Sync by design to match sdnext convention. Ships chat, list_models,
validate_key, probe_endpoints, cancel, and generate_image (with the four
image_via dispatch paths). generate_video / tts / transcribe stubs raise
NotImplementedError pending the video and audio code paths.
"""

import base64
import functools
import io
import os
import struct
import time
from pathlib import Path

import httpx
from PIL import Image
from pydantic import TypeAdapter, ValidationError  # pylint: disable=no-name-in-module

from modules import shared
from modules.json_helpers import readfile
from modules.logger import log

from modules.cloud.errors import InputValidationError, ProviderError
from modules.cloud.presets import resolve_input_limits
from modules.cloud.protocol import (
    AudioResult,
    ChatResult,
    CloudUsage,
    ImageResult,
    ProgressCallback,
    SizeConstraint,
    TranscribeResult,
    VideoResult,
)
from modules.cloud.response_models import (
    ChatResponse,
    ImageResponse,
    NanogptVideoStatus,
    NanogptVideoSubmit,
    Usage,
    VideoStatusResponse,
    VideoSubmitResponse,
)
from modules.cloud.transport import HttpTransport


# Video polling constants. Tuned for typical Sora / Kling / Pruna response cadence.
# `VIDEO_POLL_INITIAL_DELAY` avoids a guaranteed-to-be-empty first poll right after
# submission (most providers have a queue stage of at least a few seconds).
# Total wall-clock cap comes from `preset["timeouts"]["cloud_video"]`.
VIDEO_POLL_INTERVAL = 5.0
VIDEO_POLL_INITIAL_DELAY = 2.0
VIDEO_SUCCESS_STATUSES = frozenset({"completed", "succeeded"})
VIDEO_TERMINAL_STATUSES = frozenset({"completed", "succeeded", "failed", "cancelled", "canceled", "error"})


# SD_CLOUD_DEBUG=1 enables verbose per-request logging. Mirrors gallery.py:18 pattern.
debug = log.debug if os.environ.get("SD_CLOUD_DEBUG") else lambda *args, **kwargs: None


def noop_progress(_event: dict) -> None:
    """Default progress callback. Used when callers don't care about progress events."""


# ---- size_constraints.json loader -------------------------------------------

SIZE_CONSTRAINTS_PATH = Path(__file__).parent / "size_constraints.json"
SIZE_CONSTRAINTS_SCHEMA_VERSION = 1
_SIZE_CONSTRAINT_ADAPTER = TypeAdapter(SizeConstraint)


@functools.cache
def load_size_constraints() -> dict[str, SizeConstraint]:
    """Parse size_constraints.json once per process, keyed by 'provider/model'.

    Validation errors on individual entries are logged and the entry is skipped;
    a single bad entry must not prevent the rest of the catalog from loading.
    Unknown schema_version is treated as 'cannot interpret' and yields an empty
    map (callers fall through to size_constraint=None for every model).

    Underscore-prefixed keys (e.g. `_source`, `_inferred_from`) carry probe
    provenance metadata and are stripped before validation; they are
    informational only and not part of the SizeConstraint schema.
    """
    raw = readfile(str(SIZE_CONSTRAINTS_PATH), as_type="dict", silent=True)
    if not raw:
        return {}
    if raw.get("schema_version") != SIZE_CONSTRAINTS_SCHEMA_VERSION:
        log.warning(f"Cloud: size_constraints.json schema_version={raw.get('schema_version')} not supported (expected {SIZE_CONSTRAINTS_SCHEMA_VERSION}); ignoring")
        return {}
    out: dict[str, SizeConstraint] = {}
    for key, payload in raw.get("entries", {}).items():
        try:
            constraint_payload = {k: v for k, v in payload.items() if not k.startswith("_")}
            out[key] = _SIZE_CONSTRAINT_ADAPTER.validate_python(constraint_payload)
        except ValidationError as e:
            log.warning(f"Cloud: size_constraints.json entry {key} failed validation: {e}")
    return out


def get_size_constraint(provider_id: str, model_id: str) -> SizeConstraint | None:
    """Look up the constraint for a given provider+model pair.

    Returns None when the entry is absent so callers can short-circuit
    pre-flight validation without further checks.
    """
    return load_size_constraints().get(f"{provider_id}/{model_id}")


# ---- multi_image_constraints.json loader ------------------------------------

MULTI_IMAGE_CONSTRAINTS_PATH = Path(__file__).parent / "multi_image_constraints.json"
MULTI_IMAGE_CONSTRAINTS_SCHEMA_VERSION = 1


@functools.cache
def load_multi_image_constraints() -> dict[str, dict]:
    """Parse multi_image_constraints.json once per process, keyed by 'provider/model'.

    Each entry is a plain dict with two recognised keys:
      multi_image: bool       (default False)
      max_images:  int | None (default None; None means uncapped or unknown)

    Underscore-prefixed keys carry provenance metadata (e.g. probe source) and
    are stripped at load time. Unknown schema_version yields an empty map so
    callers fall through to live extraction (or to default False / None).
    """
    raw = readfile(str(MULTI_IMAGE_CONSTRAINTS_PATH), as_type="dict", silent=True)
    if not raw:
        return {}
    if raw.get("schema_version") != MULTI_IMAGE_CONSTRAINTS_SCHEMA_VERSION:
        log.warning(f"Cloud: multi_image_constraints.json schema_version={raw.get('schema_version')} not supported (expected {MULTI_IMAGE_CONSTRAINTS_SCHEMA_VERSION}); ignoring")
        return {}
    out: dict[str, dict] = {}
    for key, payload in raw.get("entries", {}).items():
        if not isinstance(payload, dict):
            log.warning(f"Cloud: multi_image_constraints.json entry {key} is not a dict; skipping")
            continue
        clean = {k: v for k, v in payload.items() if not k.startswith("_")}
        out[key] = {
            "multi_image": bool(clean.get("multi_image", False)),
            "max_images": clean.get("max_images"),
        }
    return out


def get_multi_image_constraint(provider_id: str, model_id: str) -> dict | None:
    """JSON-side override lookup for the (provider, model) pair.

    Returns the {multi_image, max_images} dict when an entry exists, else None
    so the caller can fall through to live extraction.
    """
    return load_multi_image_constraints().get(f"{provider_id}/{model_id}")


def extract_multi_image_info(raw_model: dict, supported_params: list[dict] | None) -> dict | None:
    """Live extraction of multi_image / max_images from provider metadata.

    Currently reads NanoGPT's `supported_parameters.max_images` directly off the
    raw model body. OpenRouter and OpenAI presets do not advertise; they return
    None (caller falls through to JSON override or default-False).

    `supported_params` is the already-normalised descriptor list (kept for
    future per-provider hooks); the dict-shape live data lives on `raw_model`.
    """
    _ = supported_params  # reserved for future per-provider hook
    supported = raw_model.get("supported_parameters")
    if isinstance(supported, dict):
        max_images = supported.get("max_images")
        if isinstance(max_images, int) and max_images > 0:
            return {"multi_image": max_images > 1, "max_images": max_images}
    return None


class OpenAICompatAdapter:
    """Satisfies ProviderAdapter via structural typing (sync)."""

    def __init__(self, provider_id: str, base_url: str, preset: dict, key: str):
        self.provider_id = provider_id
        self.preset = preset
        self.transport = HttpTransport(provider_id, base_url, preset, key)

    # ---- public text surface ----------------------------------------------------

    def list_models(self) -> list[dict]:
        all_models: list[dict] = []
        endpoints = self.preset.get("model_list", ["/v1/models"])
        params = self.preset.get("model_list_params", {})
        for endpoint in endpoints:
            try:
                data = self.transport.get_cached(endpoint, ttl=300, params=params or None)
                if isinstance(data, dict) and "data" in data:
                    models = data["data"]
                elif isinstance(data, list):
                    models = data
                else:
                    debug(f"Cloud: list_models unrecognized shape provider={self.provider_id} endpoint={endpoint}")
                    continue
                for model in models:
                    if isinstance(model, dict):
                        model["_source_endpoint"] = endpoint
                        all_models.append(model)
            except Exception as e:
                log.warning(f"Cloud: list_models endpoint failed provider={self.provider_id} endpoint={endpoint}: {e}")
                continue
        normalized = self.normalize_models(all_models)
        debug(f"Cloud: list_models provider={self.provider_id} raw={len(all_models)} normalized={len(normalized)}")
        return normalized

    def chat(self, params: dict, on_progress: ProgressCallback = noop_progress) -> ChatResult:
        on_progress({"type": "cloud_progress", "phase": "submitted"})
        log.info(f"Cloud: chat provider={self.provider_id} model={params.get('model')} messages={len(params.get('messages') or [])}")

        messages = params.get("messages", [])
        if not messages and params.get("prompt"):
            messages = [{"role": "user", "content": params["prompt"]}]

        body = {"model": params["model"], "messages": messages}
        chat_map = self.preset.get("param_maps", {}).get("chat", {})
        skip_keys = {"model", "messages", "prompt", "provider", "type", "priority", "extra_params"}
        for caller_name, value in params.items():
            if caller_name in skip_keys:
                continue
            mapping = chat_map.get(caller_name)
            if mapping is None:
                continue
            api_name, transform = mapping if isinstance(mapping, tuple) else (mapping, None)
            if api_name is None:
                continue
            result_val = transform(value) if transform else value
            if result_val is not None:
                body[api_name] = result_val

        if params.get("extra_params"):
            body.update(params["extra_params"])

        on_progress({"type": "cloud_progress", "phase": "processing"})
        data = self.transport.post("/v1/chat/completions", json=body)
        parsed = self.parse_response(ChatResponse, data)

        if not parsed.choices:
            return ChatResult(content="", finish_reason="stop", usage=self.parse_usage(parsed.usage))
        choice = parsed.choices[0]
        content = choice.message.content
        # Multimodal chat responses can return content as a list of parts; for the
        # text-only chat path, concatenate any text parts so callers always get a
        # string. extract_images_from_chat handles the image parts separately.
        if isinstance(content, list):
            content = "".join(part.text or "" for part in content if part.type == "text")
        return ChatResult(
            content=content or "",
            tool_calls=[tc.model_dump(exclude_unset=True) for tc in choice.message.tool_calls] if choice.message.tool_calls else None,
            finish_reason=choice.finish_reason,
            usage=self.parse_usage(parsed.usage),
        )

    def validate_key(self) -> bool:
        try:
            endpoints = self.preset.get("model_list", ["/v1/models"])
            self.transport.get(endpoints[0], params={"limit": "1"})
            log.debug(f"Cloud: validate_key ok provider={self.provider_id}")
            return True
        except Exception as e:
            log.warning(f"Cloud: validate_key failed provider={self.provider_id}: {e}")
            return False

    def probe_endpoints(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        probe_paths = [
            ("models", "/v1/models"),
            ("chat", "/v1/chat/completions"),
            ("images", "/v1/images/generations"),
            ("audio_speech", "/v1/audio/speech"),
            ("audio_transcriptions", "/v1/audio/transcriptions"),
            ("video", "/v1/videos"),
        ]
        for name, path in probe_paths:
            try:
                response = self.transport.client.request("OPTIONS", path)
                results[name] = response.status_code < 500
            except Exception as e:
                debug(f"Cloud: probe OPTIONS failed provider={self.provider_id} path={path}, falling back to GET: {e}")
                try:
                    response = self.transport.client.get(path)
                    results[name] = response.status_code != 404
                except Exception as e2:
                    debug(f"Cloud: probe GET failed provider={self.provider_id} path={path}: {e2}")
                    results[name] = False
        debug(f"Cloud: probe_endpoints provider={self.provider_id} results={results}")
        return results

    def cancel(self, remote_id: str) -> bool:
        """Best-effort cancel of a remote job.

        Bypasses the transport's request() wrapper because that wrapper
        aborts on shared.state.interrupted - and cancel() is precisely
        the cleanup we want to run AFTER an interrupt. Goes directly to
        the underlying httpx client so the POST always reaches the wire.
        """
        try:
            response = self.transport.client.post(f"/v1/videos/{remote_id}/cancel", json={})
            if response.status_code < 400:
                log.info(f"Cloud: cancelled provider={self.provider_id} remote_id={remote_id}")
                return True
            log.warning(f"Cloud: cancel returned status={response.status_code} provider={self.provider_id} remote_id={remote_id}")
            return False
        except Exception as e:
            log.warning(f"Cloud: cancel failed provider={self.provider_id} remote_id={remote_id}: {e}")
            return False

    def close(self) -> None:
        self.transport.close()

    # ---- public image surface ---------------------------------------------------

    def apply_images_transform(self, params: dict) -> dict | None:
        """Resolve a preset's images_transform hook for multi-image dispatch.

        Contract: when the caller supplies more than one reference image, the
        preset's `param_maps.images_transform` (if present) takes responsibility
        for producing the wire shape. Return shape is a dict with optional keys
        consumed by the four `_via_*` paths:

            {"json":    dict, ... }   # merged into the outgoing JSON body
            {"files":   list, ... }   # passed to httpx multipart `files=`
            {"content": list, ... }   # appended to the chat content parts

        Transforms receive the raw bytes list and own their own encoding
        (base64, dataurl, multipart tuple, provider-specific). When the
        transform is absent for a model the caller advertised as `multi_image`,
        we log a warning and return None so the four paths fall back to
        first-image-only (params["image"]). Empty / single-image requests
        short-circuit and skip the hook entirely so the original code paths
        run unchanged.
        """
        images = params.get("images")
        if not images or len(images) <= 1:
            return None
        transform = self.preset.get("param_maps", {}).get("images_transform")
        if transform is None:
            log.warning(
                f"Cloud: provider={self.provider_id} model={params.get('model')} got "
                f"{len(images)} reference images but preset has no images_transform; "
                f"falling back to first image only"
            )
            return None
        return transform(images) or {}

    def generate_image(self, params: dict, on_progress: ProgressCallback = noop_progress) -> ImageResult:
        """Dispatch to the right image_via path based on preset and has_image.

        Dispatch table:

            image_via    has_image=False        has_image=True
            "images"     via_endpoint           generate_image_edit (multipart edits)
            "dataurl"    via_endpoint           generate_image_via_dataurl
            "chat"       via_chat               via_chat (image as content part)
            "probe"      via_endpoint           via_endpoint
        """
        on_progress({"phase": "submitted"})
        has_image = bool(params.get("image"))
        image_via = self.preset.get("image_via", "images")
        log.info(f"Cloud: generate_image provider={self.provider_id} model={params.get('model')} size={params.get('width')}x{params.get('height')} n={params.get('n', 1)} has_image={has_image} via={image_via}")
        if has_image and image_via == "images":
            return self.generate_image_edit(params, on_progress)
        if has_image and image_via == "dataurl":
            return self.generate_image_via_dataurl(params, on_progress)
        if image_via == "chat":
            return self.generate_image_via_chat(params, on_progress)
        return self.generate_image_via_endpoint(params, on_progress)

    def generate_image_via_endpoint(self, params: dict, on_progress: ProgressCallback) -> ImageResult:
        """Standard /v1/images/generations JSON post (OpenAI / NanoGPT txt2img / custom)."""
        body = self.build_image_params(params)
        body["model"] = params["model"]
        multi = self.apply_images_transform(params)
        if multi and multi.get("json"):
            body.update(multi["json"])
        debug(f"Cloud: image request body provider={self.provider_id} body={body}")
        on_progress({"phase": "processing"})
        data = self.transport.post("/v1/images/generations", json=body)
        parsed = self.parse_response(ImageResponse, data)
        on_progress({"phase": "downloading"})
        images = self.extract_images(parsed)
        return ImageResult(
            images=images,
            revised_prompt=(parsed.data[0].revised_prompt if parsed.data else None),
            format="png",
            usage=self.parse_usage(parsed.usage),
        )

    def generate_image_edit(self, params: dict, on_progress: ProgressCallback) -> ImageResult:
        """Multipart /v1/images/edits (OpenAI img2img + inpaint)."""
        on_progress({"phase": "processing"})
        image_data = params["image"]
        self.validate_input_image(image_data, params["model"])
        # Multi-image edits (e.g. OpenAI gpt-image-1 with `image[]`) come through
        # apply_images_transform which returns the httpx-ready files object.
        # Single-image (or missing transform) falls back to the singular `image`.
        multi = self.apply_images_transform(params)
        files: dict | list = (
            multi["files"] if multi and multi.get("files")
            else {"image": ("input.png", image_data, "image/png")}
        )
        data_fields: dict = {"model": params["model"], "prompt": params.get("prompt", "")}
        if params.get("size"):
            data_fields["size"] = params["size"]
        if params.get("n"):
            data_fields["n"] = str(params["n"])
        if params.get("mask"):
            mask_data = self.invert_mask_for_openai(params["mask"])
            mask_entry = ("mask.png", mask_data, "image/png")
            if isinstance(files, dict):
                files["mask"] = mask_entry
            else:
                files = [*files, ("mask", mask_entry)]
        # files= triggers httpx multipart encoding with auto-generated boundary.
        # transport.build_headers() deliberately omits Content-Type so the
        # auto-detected multipart header wins (see transport.py:50-62).
        response = self.transport.client.post("/v1/images/edits", files=files, data=data_fields)
        if response.status_code >= 400:
            self.transport.raise_for_status(response)
        parsed = self.parse_response(ImageResponse, response.json())
        on_progress({"phase": "downloading"})
        images = self.extract_images(parsed)
        return ImageResult(
            images=images,
            revised_prompt=(parsed.data[0].revised_prompt if parsed.data else None),
            format="png",
            usage=self.parse_usage(parsed.usage),
        )

    def generate_image_via_dataurl(self, params: dict, on_progress: ProgressCallback) -> ImageResult:
        """img2img via imageDataUrl in the JSON body (NanoGPT pattern)."""
        on_progress({"phase": "processing"})
        image_data = params["image"]
        self.validate_input_image(image_data, params["model"])
        body = self.build_image_params(params)
        body["model"] = params["model"]
        # Multi-image dispatch: when the transform yielded a JSON payload it owns
        # the multi-reference shape (e.g. Seedream's images: [{data: ...}, ...]).
        # Otherwise build the single-image imageDataUrl as before.
        multi = self.apply_images_transform(params)
        if multi and multi.get("json"):
            body.update(multi["json"])
        else:
            fmt = self.detect_format(image_data)
            b64 = base64.b64encode(image_data).decode("ascii")
            body["imageDataUrl"] = f"data:image/{fmt};base64,{b64}"
        if params.get("mask"):
            mask_b64 = base64.b64encode(params["mask"]).decode("ascii")
            body["maskDataUrl"] = f"data:image/png;base64,{mask_b64}"
        data = self.transport.post("/v1/images/generations", json=body)
        parsed = self.parse_response(ImageResponse, data)
        on_progress({"phase": "downloading"})
        images = self.extract_images(parsed)
        return ImageResult(
            images=images,
            revised_prompt=(parsed.data[0].revised_prompt if parsed.data else None),
            format="png",
            usage=self.parse_usage(parsed.usage),
        )

    def generate_image_via_chat(self, params: dict, on_progress: ProgressCallback) -> ImageResult:
        """Image generation via /v1/chat/completions (OpenRouter / multimodal chat)."""
        prompt = params.get("prompt", "")
        content: list[dict] | str = prompt
        # Multi-image dispatch goes through the transform's "content" key, which
        # supplies the additional image_url parts. The text prompt is always
        # prepended so the chat shape stays consistent.
        multi = self.apply_images_transform(params)
        if multi and multi.get("content"):
            content = [{"type": "text", "text": prompt}, *multi["content"]]
        elif params.get("image"):
            image_data = params["image"]
            self.validate_input_image(image_data, params["model"])
            fmt = self.detect_format(image_data)
            b64 = base64.b64encode(image_data).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}},
            ]
        body: dict = {
            "model": params["model"],
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image"],
        }
        image_map = self.preset.get("param_maps", {}).get("image", {})
        skip_keys = {"model", "prompt", "provider", "type", "priority", "extra_params", "width", "height", "image", "mask"}
        for caller_name, value in params.items():
            if caller_name in skip_keys:
                continue
            mapping = image_map.get(caller_name)
            if mapping is None:
                continue
            api_name, transform = mapping if isinstance(mapping, tuple) else (mapping, None)
            if api_name is None or api_name.startswith("_"):
                continue
            result_val = transform(value) if transform else value
            if result_val is not None:
                body[api_name] = result_val
        size_fn = self.preset.get("param_maps", {}).get("image_size_transform")
        if size_fn and params.get("width") and params.get("height"):
            body.update(size_fn(params["width"], params["height"]))
        if params.get("extra_params"):
            body.update(params["extra_params"])
        on_progress({"phase": "processing"})
        data = self.transport.post("/v1/chat/completions", json=body)
        parsed = self.parse_response(ChatResponse, data)
        on_progress({"phase": "downloading"})
        images = self.extract_images_from_chat(parsed)
        return ImageResult(
            images=images,
            format="png",
            usage=self.parse_usage(parsed.usage),
        )

    # ---- video / audio stubs ----------------------------------------------------

    def generate_video(self, params: dict, on_progress: ProgressCallback = noop_progress) -> VideoResult:
        """Dispatch video generation to the right backend based on preset.video_via.

        Dispatch table:

            video_via    backend
            "videos"     /v1/videos (OpenAI Sora pattern)
            "nanogpt"    /api/generate-video + /api/video/status (NanoGPT)
            "probe"      /v1/videos as best-effort default
        """
        on_progress({"phase": "submitted"})
        has_image = bool(params.get("image"))
        video_via = self.preset.get("video_via", "videos")
        log.info(f"Cloud: generate_video provider={self.provider_id} model={params.get('model')} duration={params.get('duration')} aspect={params.get('aspect_ratio')} size={params.get('size')} has_image={has_image} via={video_via}")
        if video_via == "nanogpt":
            return self.generate_video_via_nanogpt(params, on_progress)
        return self.generate_video_via_endpoint(params, on_progress)

    def generate_video_via_endpoint(self, params: dict, on_progress: ProgressCallback) -> VideoResult:
        """OpenAI Sora pattern: POST /v1/videos, GET /v1/videos/{id}.

        Submit returns `id`, status response uses lowercase enum, video URL
        is at urls[0] / unsigned_urls[0] / video_url, falls back to
        GET /v1/videos/{id}/content.
        """
        has_image = bool(params.get("image"))
        body = self.build_video_params(params)
        body["model"] = params["model"]
        body["prompt"] = params.get("prompt", "")
        if has_image:
            image_data = params["image"]
            if not isinstance(image_data, bytes):
                raise ProviderError("Video init image must be bytes (decode base64 at the api_v1 boundary)", provider=self.provider_id)
            body["input_reference"] = base64.b64encode(image_data).decode("ascii")

        debug(f"Cloud: video submit body provider={self.provider_id} body_keys={list(body.keys())}")
        on_progress({"phase": "processing", "progress": 0.0})
        submit_data = self.transport.post("/v1/videos", json=body)
        submit = self.parse_response(VideoSubmitResponse, submit_data)
        on_progress({"phase": "queued_remote", "remote_id": submit.id, "progress": 0.0})
        return self.poll_video_job(submit.id, on_progress)

    def generate_video_via_nanogpt(self, params: dict, on_progress: ProgressCallback) -> VideoResult:
        """NanoGPT pattern: POST /api/generate-video, GET /api/video/status?requestId=...

        Per docs.nano-gpt.com/api-reference/video-generation. Body uses
        duration as a string (some NanoGPT models reject numeric duration);
        we coerce to str for compatibility.
        """
        has_image = bool(params.get("image"))
        body = self.build_video_params(params)
        body["model"] = params["model"]
        body["prompt"] = params.get("prompt", "")
        # NanoGPT documents `duration` as a string; coerce if numeric came through.
        if "duration" in body and not isinstance(body["duration"], str):
            body["duration"] = str(int(body["duration"]) if float(body["duration"]).is_integer() else body["duration"])
        if has_image:
            image_data = params["image"]
            if not isinstance(image_data, bytes):
                raise ProviderError("Video init image must be bytes (decode base64 at the api_v1 boundary)", provider=self.provider_id)
            # NanoGPT's i2v body shape isn't documented; using input_reference
            # as the closest convention. May need adjustment after a live i2v test.
            body["input_reference"] = base64.b64encode(image_data).decode("ascii")

        debug(f"Cloud: nanogpt video submit body provider={self.provider_id} body_keys={list(body.keys())}")
        on_progress({"phase": "processing", "progress": 0.0})
        # NanoGPT base_url already ends in /api; submit path is /generate-video.
        submit_data = self.transport.post("/generate-video", json=body)
        submit = self.parse_response(NanogptVideoSubmit, submit_data)
        on_progress({"phase": "queued_remote", "remote_id": submit.id, "progress": 0.0})
        return self.poll_video_via_nanogpt(submit.id, on_progress)

    def poll_video_via_nanogpt(self, video_id: str, on_progress: ProgressCallback) -> VideoResult:
        """Poll NanoGPT's /api/video/status endpoint until terminal status."""
        timeout = float(self.preset.get("timeouts", {}).get("cloud_video", 600))
        if not self.transport.sleep_interruptible(VIDEO_POLL_INITIAL_DELAY):
            raise ProviderError("Interrupted by user before first video poll", provider=self.provider_id)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if shared.state.interrupted:
                # NanoGPT does not document a cancel endpoint - just abort polling
                raise ProviderError("Interrupted by user during video poll", provider=self.provider_id)
            # NanoGPT base_url already ends in /api; status path is /video/status.
            status_data = self.transport.get(f"/video/status?requestId={video_id}")
            parsed = self.parse_response(NanogptVideoStatus, status_data)
            # NanoGPT nests everything under `data`; pull status from there.
            inner = parsed.data
            raw_status = (inner.status if inner else "") or ""
            status_lower = raw_status.lower()
            raw_progress = inner.progress if inner else None
            progress = self.normalize_progress(raw_progress)
            debug(f"Cloud: nanogpt video poll provider={self.provider_id} remote_id={video_id} status={status_lower} progress={progress} elapsed={time.time() - t0:.1f}s")
            on_progress({"phase": "processing", "progress": progress, "remote_status": status_lower})

            if status_lower in {"completed", "succeeded"}:
                on_progress({"phase": "downloading", "progress": 0.95, "remote_status": status_lower})
                video_url = self.extract_nanogpt_video_url(parsed)
                if not video_url:
                    raise ProviderError(f"NanoGPT video {video_id} reported {status_lower} but no URL in response", provider=self.provider_id)
                video_bytes = self.download_url(video_url)
                if not video_bytes:
                    raise ProviderError(f"Failed to download NanoGPT video {video_id} from {video_url[:120]}", provider=self.provider_id)
                log.info(f"Cloud: video completed provider={self.provider_id} remote_id={video_id} bytes={len(video_bytes)} elapsed={time.time() - t0:.1f}s")
                return VideoResult(
                    data=video_bytes,
                    format="mp4",
                    duration=(inner.duration if inner else None),
                    usage=self.parse_usage(parsed.usage),
                )
            if status_lower in {"failed", "error", "cancelled", "canceled"}:
                err_field = (inner.error if inner else None)
                err = err_field if isinstance(err_field, (str, dict)) else f"status={raw_status}"
                if isinstance(err, dict):
                    err = err.get("message") or err.get("code") or str(err)
                log.warning(f"Cloud: nanogpt video terminal-non-success provider={self.provider_id} remote_id={video_id} status={status_lower} err={err!r}")
                raise ProviderError(f"Video {status_lower}: {err}", provider=self.provider_id)
            if not self.transport.sleep_interruptible(VIDEO_POLL_INTERVAL):
                raise ProviderError("Interrupted by user during video poll backoff", provider=self.provider_id)

        log.error(f"Cloud: nanogpt video timeout provider={self.provider_id} remote_id={video_id} elapsed={time.time() - t0:.1f}s")
        raise ProviderError(f"Video generation timed out after {timeout}s", provider=self.provider_id)

    @staticmethod
    def extract_nanogpt_video_url(status: NanogptVideoStatus) -> str | None:
        """Pull the video URL out of NanoGPT's nested response structure.

        Tries data.output.video.url first (Pruna shape), then falls back to
        data.output.videoUrls[0] (some other Pruna routes return both forms).
        """
        if status.data and status.data.output:
            video = status.data.output.video
            if isinstance(video, dict):
                url = video.get("url")
                if isinstance(url, str) and url:
                    return url
            urls = status.data.output.videoUrls
            if isinstance(urls, list) and urls and isinstance(urls[0], str):
                return urls[0]
        return None

    def poll_video_job(self, video_id: str, on_progress: ProgressCallback) -> VideoResult:
        """Poll /v1/videos/{id} until terminal status, then download the bytes."""
        timeout = float(self.preset.get("timeouts", {}).get("cloud_video", 600))
        # Initial delay; honour interrupt even during this short sleep.
        if not self.transport.sleep_interruptible(VIDEO_POLL_INITIAL_DELAY):
            self.cancel(video_id)
            raise ProviderError("Interrupted by user before first video poll", provider=self.provider_id)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if shared.state.interrupted:
                self.cancel(video_id)
                raise ProviderError("Interrupted by user during video poll", provider=self.provider_id)
            status_data = self.transport.get(f"/v1/videos/{video_id}")
            parsed = self.parse_response(VideoStatusResponse, status_data)
            status = (parsed.status or "").lower()
            progress = self.normalize_progress(parsed.progress)
            debug(f"Cloud: video poll provider={self.provider_id} remote_id={video_id} status={status} progress={progress} elapsed={time.time() - t0:.1f}s")
            on_progress({"phase": "processing", "progress": progress, "remote_status": status})

            if status in VIDEO_SUCCESS_STATUSES:
                on_progress({"phase": "downloading", "progress": 0.95, "remote_status": status})
                video_bytes = self.download_video_content(video_id, parsed)
                if not video_bytes:
                    raise ProviderError(f"Video {video_id} reported {status} but no bytes returned", provider=self.provider_id)
                log.info(f"Cloud: video completed provider={self.provider_id} remote_id={video_id} bytes={len(video_bytes)} elapsed={time.time() - t0:.1f}s")
                return VideoResult(
                    data=video_bytes,
                    format="mp4",
                    duration=parsed.seconds if parsed.seconds is not None else parsed.duration,
                    usage=self.parse_usage(parsed.usage),
                )

            if status in VIDEO_TERMINAL_STATUSES:
                err = self.extract_video_error(parsed)
                log.warning(f"Cloud: video terminal-non-success provider={self.provider_id} remote_id={video_id} status={status} err={err!r}")
                raise ProviderError(f"Video {status}: {err}", provider=self.provider_id)

            if not self.transport.sleep_interruptible(VIDEO_POLL_INTERVAL):
                self.cancel(video_id)
                raise ProviderError("Interrupted by user during video poll backoff", provider=self.provider_id)

        # Timed out
        log.error(f"Cloud: video timeout provider={self.provider_id} remote_id={video_id} elapsed={time.time() - t0:.1f}s")
        self.cancel(video_id)
        raise ProviderError(f"Video generation timed out after {timeout}s", provider=self.provider_id)

    def download_video_content(self, video_id: str, status: VideoStatusResponse) -> bytes | None:
        """Resolve the video bytes from the provider response.

        Precedence:
            1. status.urls[0] (signed URL list, OpenAI Sora)
            2. status.unsigned_urls[0] (legacy, some Kling routes)
            3. status.video_url (singular field, observed in some custom providers)
            4. Fallback: GET /v1/videos/{id}/content (Sora's content streaming endpoint)
        """
        candidates: list[str] = []
        candidates.extend(status.urls or [])
        candidates.extend(status.unsigned_urls or [])
        if status.video_url:
            candidates.append(status.video_url)
        for url in candidates:
            if url:
                data = self.download_url(url)
                if data:
                    return data
        # Fallback to provider's content endpoint.
        try:
            response = self.transport.client.get(f"/v1/videos/{video_id}/content")
            if response.status_code == 200:
                debug(f"Cloud: video content fallback ok provider={self.provider_id} remote_id={video_id} bytes={len(response.content)}")
                return response.content
            log.warning(f"Cloud: video content fallback status={response.status_code} provider={self.provider_id} remote_id={video_id}")
        except Exception as e:
            log.warning(f"Cloud: video content fallback failed provider={self.provider_id} remote_id={video_id}: {e}")
        return None

    def build_video_params(self, params: dict) -> dict:
        """Apply the preset's video param map to caller params.

        Mirrors build_image_params: skip a fixed set of caller-only keys, look
        up each remaining key in the preset's video param_map, apply the
        optional transform, drop None results, merge extra_params last.
        """
        video_map = self.preset.get("param_maps", {}).get("video", {})
        api_params: dict = {}
        skip_keys = {"provider", "model", "prompt", "type", "priority", "extra_params", "image"}
        for caller_name, value in params.items():
            if caller_name in skip_keys:
                continue
            mapping = video_map.get(caller_name)
            if mapping is None:
                continue
            api_name, transform = mapping if isinstance(mapping, tuple) else (mapping, None)
            if api_name is None or api_name.startswith("_"):
                continue
            result_val = transform(value) if transform else value
            if result_val is not None:
                api_params[api_name] = result_val
        if params.get("extra_params"):
            api_params.update(params["extra_params"])
        return api_params

    @staticmethod
    def normalize_progress(raw: float | int | None) -> float | None:
        """Normalize a provider's progress field to a 0-1 float.

        Providers report progress as 0-1 (Sora) or 0-100 (Kling). Heuristic:
        if raw > 1.5, treat as 0-100; else treat as 0-1. Clamps to [0, 1].
        None passes through.
        """
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if value > 1.5:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def extract_video_error(status: VideoStatusResponse) -> str:
        """Pull a human-readable message out of the provider's error envelope.

        Providers vary: some return `error: "string"`, others `error: {message: ..., code: ...}`.
        """
        err = status.error
        if isinstance(err, dict):
            return err.get("message") or err.get("code") or str(err)
        if isinstance(err, str):
            return err
        return f"status={status.status}"

    def tts(self, params: dict) -> AudioResult:
        raise NotImplementedError("Cloud audio TTS not yet implemented in this adapter")

    def transcribe(self, params: dict) -> TranscribeResult:
        raise NotImplementedError("Cloud audio STT not yet implemented in this adapter")

    # ---- image helpers ----------------------------------------------------------

    def build_image_params(self, params: dict) -> dict:
        """Apply the preset's image param map to caller params, plus image_size_transform + extra_params."""
        image_map = self.preset.get("param_maps", {}).get("image", {})
        api_params: dict = {}
        skip_keys = {"provider", "model", "type", "priority", "extra_params", "width", "height", "image", "mask"}
        for caller_name, value in params.items():
            if caller_name in skip_keys:
                continue
            mapping = image_map.get(caller_name)
            if mapping is None:
                continue
            api_name, transform = mapping if isinstance(mapping, tuple) else (mapping, None)
            # _-prefixed api names (e.g. "_message_content" for openrouter chat-image)
            # are sentinels meaning "consumed elsewhere" - skip them here.
            if api_name is None or api_name.startswith("_"):
                continue
            result_val = transform(value) if transform else value
            if result_val is not None:
                api_params[api_name] = result_val
        size_fn = self.preset.get("param_maps", {}).get("image_size_transform")
        if size_fn and params.get("width") and params.get("height"):
            api_params.update(size_fn(params["width"], params["height"]))
        if params.get("extra_params"):
            api_params.update(params["extra_params"])
        return api_params

    def extract_images(self, response: ImageResponse) -> list[bytes]:
        images: list[bytes] = []
        for item in response.data:
            if item.b64_json:
                images.append(base64.b64decode(item.b64_json))
            elif item.url:
                img_bytes = self.download_url(item.url)
                if img_bytes:
                    images.append(img_bytes)
        return images

    def extract_images_from_chat(self, response: ChatResponse) -> list[bytes]:
        images: list[bytes] = []
        for choice in response.choices:
            content = choice.message.content
            if not isinstance(content, list):
                continue
            for part in content:
                if part.type == "image_url" and part.image_url:
                    url = part.image_url.url
                    if url.startswith("data:") and "," in url:
                        b64 = url.split(",", 1)[1]
                        if b64:
                            images.append(base64.b64decode(b64))
                elif part.type == "image":
                    b64 = part.data or part.b64_json
                    if b64:
                        images.append(base64.b64decode(b64))
        return images

    def parse_response(self, model_cls, data):
        """Validate a raw provider response dict against a Pydantic model.

        On schema failure, raises ProviderError with the first validation
        error message - surfaces shape drift early at the boundary rather
        than producing KeyErrors / TypeErrors deep in extraction code.
        """
        try:
            return model_cls.model_validate(data)
        except ValidationError as e:
            errors = e.errors()
            first = errors[0] if errors else {}
            loc = ".".join(str(x) for x in first.get("loc", ()))
            msg = first.get("msg", str(e))
            raise ProviderError(
                f"Malformed {model_cls.__name__} from provider (at {loc!r}: {msg})",
                provider=self.provider_id,
            ) from e

    def download_url(self, url: str) -> bytes | None:
        """Fetch image bytes when a provider returns a URL instead of base64
        (e.g. DALL-E without response_format=b64_json). Uses a fresh client so
        absolute URLs resolve correctly outside the provider's base_url scope."""
        try:
            with httpx.Client(timeout=60) as client:
                response = client.get(url)
                if response.status_code == 200:
                    debug(f"Cloud: download ok provider={self.provider_id} bytes={len(response.content)}")
                    return response.content
                log.warning(f"Cloud: download status={response.status_code} provider={self.provider_id} url={url[:120]}")
        except Exception as e:
            log.warning(f"Cloud: download failed provider={self.provider_id} url={url[:120]}: {e}")
        return None

    def validate_input_image(self, image_data: bytes, model_id: str) -> None:
        """Validate image bytes against provider limits.

        Raises InputValidationError (HTTP 400) on violations; these are
        caller-input failures, not provider failures.
        """
        limits = resolve_input_limits(self.preset, model_id)
        max_bytes = limits.get("max_image_bytes")
        if max_bytes and len(image_data) > max_bytes:
            size_mb = len(image_data) / 1_000_000
            limit_mb = max_bytes / 1_000_000
            raise InputValidationError(
                f"Image too large ({size_mb:.1f} MB). Provider limit is {limit_mb:.1f} MB.",
                provider=self.provider_id,
                field="image",
                limit=max_bytes,
            )
        max_side = limits.get("max_longest_side")
        if max_side:
            dims = self.read_image_dimensions(image_data)
            if dims and max(dims) > max_side:
                raise InputValidationError(
                    f"Image dimensions {dims[0]}x{dims[1]} exceed provider limit of {max_side}px on longest side.",
                    provider=self.provider_id,
                    field="image",
                    limit=max_side,
                )
        allowed_formats = limits.get("formats")
        if allowed_formats:
            fmt = self.detect_format(image_data)
            if fmt not in allowed_formats:
                raise InputValidationError(
                    f"Image format {fmt!r} not supported. Provider accepts: {', '.join(allowed_formats)}.",
                    provider=self.provider_id,
                    field="image",
                    limit=allowed_formats,
                )

    def invert_mask_for_openai(self, mask_bytes: bytes) -> bytes:
        """Convert white-on-black mask (caller convention) to RGBA-alpha-0 mask
        (OpenAI /v1/images/edits convention - alpha=0 marks editable regions)."""
        img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        rgba = Image.new("RGBA", img.size, (0, 0, 0, 255))
        alpha = img.point(lambda v: 0 if v > 128 else 255)
        rgba.putalpha(alpha)
        buf = io.BytesIO()
        rgba.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def detect_format(image_data: bytes) -> str:
        """Sniff PNG / JPEG / WEBP from header. Defaults to png."""
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if image_data[:2] == b"\xff\xd8":
            return "jpeg"
        if len(image_data) >= 12 and image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return "webp"
        return "png"

    @staticmethod
    def read_image_dimensions(data: bytes) -> tuple[int, int] | None:
        """Read width/height from PNG or JPEG header without decoding the image."""
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            w, h = struct.unpack(">II", data[16:24])
            return (w, h)
        if data[:2] == b"\xff\xd8":
            offset = 2
            while offset < len(data) - 8:
                if data[offset] != 0xFF:
                    break
                marker = data[offset + 1]
                length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
                if marker in (0xC0, 0xC2):
                    h = struct.unpack(">H", data[offset + 5:offset + 7])[0]
                    w = struct.unpack(">H", data[offset + 7:offset + 9])[0]
                    return (w, h)
                offset += 2 + length
        return None

    # ---- model normalization -----------------------------------------------------

    def normalize_models(self, raw_models: list[dict]) -> list[dict]:
        from modules.cloud.codify import codify_from_model
        normalized: list[dict] = []
        for m in raw_models:
            model_id = m.get("id", "")
            if not model_id:
                continue
            modalities = self.infer_modalities(m)
            capabilities = self.infer_capabilities(m)
            pricing = self.extract_pricing(m)
            supported_params = self.extract_supported_params(m) or []

            # Resolution order: JSON override wins, then live
            # codify-from-metadata, then None. JSON normally holds only the
            # OpenAI hardcoded enums (since OpenAI's /v1/models doesn't carry
            # resolutions); NanoGPT and other rich-metadata providers get
            # constraints auto-extracted from each list_models response so they
            # stay current without re-running a discovery sweep.
            size_constraint = get_size_constraint(self.provider_id, model_id) or codify_from_model(m, supported_params)

            # Back-compat: when supported_params doesn't already advertise a
            # size enum and the new size_constraint provides equivalent data,
            # derive the enum so legacy consumers reading supported_params.size
            # keep working until they migrate.
            has_size_enum = any(p.get("name") == "size" for p in supported_params)
            if not has_size_enum and size_constraint is not None and size_constraint.kind in ("enum", "bucket") and size_constraint.options:
                supported_params.append({
                    "name": "size",
                    "type": "enum",
                    "options": size_constraint.options,
                    "default": size_constraint.default or size_constraint.options[0],
                })
            elif not has_size_enum:
                pricing_resolutions = self.extract_resolutions_from_pricing(m)
                if pricing_resolutions:
                    supported_params.append({
                        "name": "size",
                        "type": "enum",
                        "options": pricing_resolutions,
                        "default": pricing_resolutions[0],
                    })

            # Multi-image capability resolution. Same precedence as
            # size_constraint: JSON override wins, then live extraction
            # (NanoGPT advertises max_images natively), then (False, None)
            # default. Surfaced as two flat keys (no nested constraint object
            # since the shape is just two scalars).
            multi_info = get_multi_image_constraint(self.provider_id, model_id) or extract_multi_image_info(m, supported_params) or {}
            multi_image = bool(multi_info.get("multi_image", False))
            max_images = multi_info.get("max_images")

            normalized.append({
                "source": "cloud",
                "id": model_id,
                "name": m.get("name") or model_id.split("/")[-1],
                "provider": self.provider_id,
                "modalities": modalities,
                "capabilities": capabilities,
                "pricing": pricing,
                "context_length": m.get("context_length") or m.get("max_model_len"),
                "supported_params": supported_params or None,
                "description": m.get("description"),
                "default_params": m.get("default_parameters"),
                "size_constraint": size_constraint,
                "multi_image": multi_image,
                "max_images": max_images,
            })
        return normalized

    def infer_modalities(self, m: dict) -> list[str]:
        modalities: list[str] = []
        arch = m.get("architecture", {})
        output_mods = arch.get("output_modalities", [])
        input_mods = arch.get("input_modalities", [])
        if "image" in output_mods:
            modalities.append("text-to-image")
        if "image" in input_mods and "image" in output_mods:
            modalities.append("image-to-image")
        if "text" in output_mods:
            modalities.append("chat")
        if "image" in input_mods and "text" in output_mods:
            modalities.append("vision")
        if "audio" in output_mods:
            modalities.append("audio-out")
        if "audio" in input_mods:
            modalities.append("audio-in")
        if "video" in output_mods:
            modalities.append("text-to-video")
        if "video" in output_mods and "image" in input_mods:
            modalities.append("image-to-video")
        if not modalities:
            modalities.append("chat")
        return modalities

    def infer_capabilities(self, m: dict) -> list[str]:
        caps: list[str] = []
        supported = m.get("supported_parameters", [])
        if isinstance(supported, dict):
            caps.append("streaming")
            return caps
        if "tools" in supported:
            caps.append("tools")
        if "structured_outputs" in supported or "response_format" in supported:
            caps.append("structured-output")
        if "seed" in supported:
            caps.append("seed")
        if "reasoning" in supported or "reasoning_effort" in supported:
            caps.append("reasoning")
        caps.append("streaming")
        return caps

    def extract_pricing(self, m: dict) -> dict | None:
        pricing = m.get("pricing")
        if not pricing or not isinstance(pricing, dict):
            return None
        result: dict = {"currency": "USD"}
        if pricing.get("prompt"):
            result["prompt_token"] = pricing["prompt"]
        if pricing.get("completion"):
            result["completion_token"] = pricing["completion"]
        if pricing.get("image"):
            result["per_image"] = pricing["image"]
        if pricing.get("request"):
            result["per_request"] = pricing["request"]
        if pricing.get("input_cache_read"):
            result["cache_read_token"] = pricing["input_cache_read"]
        if pricing.get("input_cache_write"):
            result["cache_write_token"] = pricing["input_cache_write"]
        return result if len(result) > 1 else None

    def extract_supported_params(self, m: dict) -> list[dict] | None:
        supported = m.get("supported_parameters")
        if not supported:
            return None
        if isinstance(supported, dict):
            return self.extract_supported_params_dict(supported)
        if isinstance(supported, list):
            return self.extract_supported_params_list(supported)
        return None

    def extract_supported_params_dict(self, supported: dict) -> list[dict] | None:
        # NanoGPT format: {"resolutions": [...], "max_images": 4, ...}
        descriptors: list[dict] = []
        resolutions = supported.get("resolutions")
        if resolutions and isinstance(resolutions, list):
            options = [r.replace("*", "x") for r in resolutions]
            descriptors.append({
                "name": "size",
                "type": "enum",
                "options": options,
                "default": options[0],
            })
        return descriptors or None

    def extract_supported_params_list(self, supported: list) -> list[dict] | None:
        # OpenRouter format: ["temperature", "top_p", ...]
        descriptors: list[dict] = []
        param_schemas = {
            "temperature": {"type": "float", "min": 0.0, "max": 2.0, "step": 0.1, "default": 1.0},
            "top_p": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0},
            "top_k": {"type": "int", "min": 1, "max": 200, "default": 50},
            "max_tokens": {"type": "int", "min": 1, "max": 128000},
            "max_completion_tokens": {"type": "int", "min": 1, "max": 128000},
            "frequency_penalty": {"type": "float", "min": -2.0, "max": 2.0, "step": 0.1, "default": 0.0},
            "presence_penalty": {"type": "float", "min": -2.0, "max": 2.0, "step": 0.1, "default": 0.0},
            "repetition_penalty": {"type": "float", "min": 0.0, "max": 2.0, "step": 0.1, "default": 1.0},
            "seed": {"type": "int", "min": -1, "max": 2147483647},
        }
        for param_name in supported:
            schema = param_schemas.get(param_name)
            if schema:
                descriptors.append({"name": param_name, **schema})
            else:
                descriptors.append({"name": param_name, "type": "string"})
        return descriptors or None

    def extract_resolutions_from_pricing(self, m: dict) -> list[str] | None:
        # Fallback: infer supported sizes from pricing keys like {"1024*1024": 0.017}
        pricing = m.get("pricing")
        if not pricing or not isinstance(pricing, dict):
            return None
        per_image = pricing.get("per_image")
        if not isinstance(per_image, dict):
            return None
        resolutions: list[str] = []
        for key in per_image:
            normalized = key.replace("*", "x")
            parts = normalized.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                resolutions.append(normalized)
        return resolutions or None

    def parse_usage(self, usage: Usage | dict | None) -> CloudUsage | None:
        """Translate the upstream Usage shape into our internal CloudUsage type.

        Accepts the Pydantic Usage model (from a parsed response) or a raw dict
        (when called from model normalization paths that have not yet parsed).
        """
        if not usage:
            return None
        if isinstance(usage, dict):
            try:
                usage = Usage.model_validate(usage)
            except ValidationError:
                return None
        return CloudUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=usage.cost,
        )
