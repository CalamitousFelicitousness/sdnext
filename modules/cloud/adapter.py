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
import io
import os
import struct

import httpx
from PIL import Image

from modules.logger import log

from modules.cloud.errors import InputValidationError
from modules.cloud.presets import resolve_input_limits
from modules.cloud.protocol import (
    AudioResult,
    ChatResult,
    CloudUsage,
    ImageResult,
    ProgressCallback,
    TranscribeResult,
    VideoResult,
)
from modules.cloud.transport import HttpTransport


# SD_CLOUD_DEBUG=1 enables verbose per-request logging. Mirrors gallery.py:18 pattern.
debug = log.debug if os.environ.get("SD_CLOUD_DEBUG") else lambda *args, **kwargs: None


def noop_progress(_event: dict) -> None:
    """Default progress callback. Used when callers don't care about progress events."""


class OpenAICompatAdapter:
    """Satisfies ProviderAdapter via structural typing (sync)."""

    # Models with fixed size enums that providers don't always advertise via
    # supported_parameters. Used to populate the size constraint when missing.
    IMAGE_SIZE_CONSTRAINTS: dict[str, list[str]] = {
        "dall-e-2": ["256x256", "512x512", "1024x1024"],
        "dall-e-3": ["1024x1024", "1024x1792", "1792x1024"],
        "gpt-image-1": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "gpt-image-1-mini": ["1024x1024", "1536x1024", "1024x1536", "auto"],
    }

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

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return ChatResult(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls"),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=self.parse_usage(data.get("usage")),
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
        try:
            self.transport.post(f"/v1/videos/{remote_id}/cancel", json={})
            log.info(f"Cloud: cancelled provider={self.provider_id} remote_id={remote_id}")
            return True
        except Exception as e:
            log.warning(f"Cloud: cancel failed provider={self.provider_id} remote_id={remote_id}: {e}")
            return False

    def close(self) -> None:
        self.transport.close()

    # ---- public image surface ---------------------------------------------------

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
        debug(f"Cloud: image request body provider={self.provider_id} body={body}")
        on_progress({"phase": "processing"})
        data = self.transport.post("/v1/images/generations", json=body)
        on_progress({"phase": "downloading"})
        images = self.extract_images(data)
        return ImageResult(
            images=images,
            revised_prompt=(data.get("data", [{}])[0].get("revised_prompt") if data.get("data") else None),
            format="png",
            usage=self.parse_usage(data.get("usage")),
        )

    def generate_image_edit(self, params: dict, on_progress: ProgressCallback) -> ImageResult:
        """Multipart /v1/images/edits (OpenAI img2img + inpaint)."""
        on_progress({"phase": "processing"})
        image_data = params["image"]
        self.validate_input_image(image_data, params["model"])
        files: dict = {"image": ("input.png", image_data, "image/png")}
        data_fields: dict = {"model": params["model"], "prompt": params.get("prompt", "")}
        if params.get("size"):
            data_fields["size"] = params["size"]
        if params.get("n"):
            data_fields["n"] = str(params["n"])
        if params.get("mask"):
            mask_data = self.invert_mask_for_openai(params["mask"])
            files["mask"] = ("mask.png", mask_data, "image/png")
        # Strip Content-Type so httpx sets the multipart boundary itself.
        headers = {k: v for k, v in self.transport.client.headers.items() if k.lower() != "content-type"}
        response = self.transport.client.post("/v1/images/edits", files=files, data=data_fields, headers=headers)
        if response.status_code >= 400:
            self.transport.raise_for_status(response)
        result_data = response.json()
        on_progress({"phase": "downloading"})
        images = self.extract_images(result_data)
        return ImageResult(
            images=images,
            revised_prompt=(result_data.get("data", [{}])[0].get("revised_prompt") if result_data.get("data") else None),
            format="png",
            usage=self.parse_usage(result_data.get("usage")),
        )

    def generate_image_via_dataurl(self, params: dict, on_progress: ProgressCallback) -> ImageResult:
        """img2img via imageDataUrl in the JSON body (NanoGPT pattern)."""
        on_progress({"phase": "processing"})
        image_data = params["image"]
        self.validate_input_image(image_data, params["model"])
        fmt = self.detect_format(image_data)
        b64 = base64.b64encode(image_data).decode("ascii")
        body = self.build_image_params(params)
        body["model"] = params["model"]
        body["imageDataUrl"] = f"data:image/{fmt};base64,{b64}"
        if params.get("mask"):
            mask_b64 = base64.b64encode(params["mask"]).decode("ascii")
            body["maskDataUrl"] = f"data:image/png;base64,{mask_b64}"
        data = self.transport.post("/v1/images/generations", json=body)
        on_progress({"phase": "downloading"})
        images = self.extract_images(data)
        return ImageResult(
            images=images,
            revised_prompt=(data.get("data", [{}])[0].get("revised_prompt") if data.get("data") else None),
            format="png",
            usage=self.parse_usage(data.get("usage")),
        )

    def generate_image_via_chat(self, params: dict, on_progress: ProgressCallback) -> ImageResult:
        """Image generation via /v1/chat/completions (OpenRouter / multimodal chat)."""
        prompt = params.get("prompt", "")
        content: list[dict] | str = prompt
        if params.get("image"):
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
        on_progress({"phase": "downloading"})
        images = self.extract_images_from_chat(data)
        return ImageResult(
            images=images,
            format="png",
            usage=self.parse_usage(data.get("usage")),
        )

    # ---- video / audio stubs ----------------------------------------------------

    def generate_video(self, params: dict, on_progress: ProgressCallback = noop_progress) -> VideoResult:
        raise NotImplementedError("Cloud video generation not yet implemented in this adapter")

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

    def extract_images(self, data: dict) -> list[bytes]:
        images: list[bytes] = []
        for item in data.get("data", []):
            if item.get("b64_json"):
                images.append(base64.b64decode(item["b64_json"]))
            elif item.get("url"):
                img_bytes = self.download_url(item["url"])
                if img_bytes:
                    images.append(img_bytes)
        return images

    def extract_images_from_chat(self, data: dict) -> list[bytes]:
        images: list[bytes] = []
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    url_data = part.get("image_url", {}).get("url", "")
                    if url_data.startswith("data:") and "," in url_data:
                        b64 = url_data.split(",", 1)[1]
                        if b64:
                            images.append(base64.b64decode(b64))
                elif part.get("type") == "image":
                    b64 = part.get("data", "") or part.get("b64_json", "")
                    if b64:
                        images.append(base64.b64decode(b64))
        return images

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
        normalized: list[dict] = []
        for m in raw_models:
            model_id = m.get("id", "")
            if not model_id:
                continue
            modalities = self.infer_modalities(m)
            capabilities = self.infer_capabilities(m)
            pricing = self.extract_pricing(m)
            supported_params = self.extract_supported_params(m) or []

            has_size_enum = any(p.get("name") == "size" for p in supported_params)
            if not has_size_enum:
                size_options = self.get_size_constraints(model_id) or self.extract_resolutions_from_pricing(m)
                if size_options:
                    supported_params.append({
                        "name": "size",
                        "type": "enum",
                        "options": size_options,
                        "default": size_options[0],
                    })

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
            })
        return normalized

    def get_size_constraints(self, model_id: str) -> list[str] | None:
        bare_id = model_id.split("/")[-1] if "/" in model_id else model_id
        for family, sizes in self.IMAGE_SIZE_CONSTRAINTS.items():
            if bare_id == family or bare_id.startswith(f"{family}:"):
                return sizes
        return None

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

    def parse_usage(self, usage_data: dict | None) -> CloudUsage | None:
        if not usage_data:
            return None
        return CloudUsage(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            cost=usage_data.get("cost"),
        )
