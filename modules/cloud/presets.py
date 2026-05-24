"""Built-in cloud provider presets and the resolve_input_limits helper.

Each preset encodes provider-specific endpoint routing, header config, and
per-modality parameter mapping. Consumed by transport.py (auth_header,
extra_headers, model_list, timeouts) and adapter.py (image_via, param_maps,
input_limits).

Chat and image generation consume the chat / auth / model_list / image
portions today; audio and video parameter maps are carried unchanged so the
video and audio code paths can plug straight in.

Each preset's param_maps may include an optional `images_transform` lambda.
The transform receives the raw bytes list (already validated for size /
format upstream by the adapter) and returns a dict whose optional keys are
consumed by the matching `_via_*` dispatch path:

    {"json":    dict, ...}    # merged into the outgoing JSON body
    {"files":   list, ...}    # passed to httpx multipart `files=`
    {"content": list, ...}    # appended to chat content parts

Transforms are *capability advertisements*: a preset that defines one
declares that its provider can accept multi-image input on the dispatch
path implied by `image_via`. Models within that preset still need
`multi_image=true` (from live extraction or JSON override) for the
preflight to clear. When a multi-image request reaches a preset without
an `images_transform`, the adapter degrades to first-image-only and logs.
"""


from modules.cloud.encoding import detect_image_format, to_dataurl


DEFAULT_CHAT_PARAMS = {
    "prompt": ("prompt", None),
    "temperature": ("temperature", lambda v: max(0.0, min(2.0, v))),
    "top_p": ("top_p", lambda v: max(0.0, min(1.0, v))),
    "max_tokens": ("max_tokens", None),
    "seed": ("seed", lambda v: v if v >= 0 else None),
    "stop": ("stop", None),
}

DEFAULT_IMAGE_PARAMS = {
    "prompt": ("prompt", None),
    "negative_prompt": ("negative_prompt", None),
    "guidance": ("guidance_scale", lambda v: max(1.0, min(20.0, v))),
    "steps": ("num_inference_steps", lambda v: max(1, min(150, v))),
    "seed": ("seed", lambda v: v if v >= 0 else None),
    "n": ("n", lambda v: max(1, min(10, v))),
    "strength": ("strength", None),
    "size": ("size", None),
    "quality": ("quality", None),
    "style": ("style", None),
}

DEFAULT_TTS_PARAMS = {
    "input": ("input", None),
    "voice": ("voice", None),
    "speed": ("speed", lambda v: max(0.25, min(4.0, v))),
    "response_format": ("response_format", None),
}

# Video param map. Most cloud video providers (Kling, Pruna, Wavespeed,
# AIHubMix passthrough) accept duration / aspect_ratio / size with the names
# below. Sora 2 differs (`seconds` / `orientation` / `resolution`); callers
# targeting Sora-specific models pass equivalents via `extra_params` until a
# per-model preset family lands. Negative seed dropped (None means random).
DEFAULT_VIDEO_PARAMS = {
    "prompt": ("prompt", None),
    "negative_prompt": ("negative_prompt", None),
    "duration": ("duration", lambda v: max(0.1, min(60.0, float(v)))),
    "aspect_ratio": ("aspect_ratio", None),
    "size": ("size", None),
    "seed": ("seed", lambda v: v if v >= 0 else None),
    "n": ("n", lambda v: max(1, min(4, v))),
}


def size_transform_wxh(w: int, h: int) -> dict:
    """Convert width/height to the WxH string format most providers expect."""
    return {"size": f"{w}x{h}"}


# These transforms are drafted from provider docs; the live probe sweep
# verifies and corrects each shape per priority model. Each transform
# receives the raw bytes list and returns the dispatch-path-specific
# payload (see module docstring).


def nanogpt_images_transform(images: list[bytes]) -> dict:
    """NanoGPT dataurl-path multi-image: plural of the single-image field.

    Single-image (existing): `imageDataUrl: "data:image/...;base64,..."`.
    Multi-image (drafted from the singular-vs-plural convention): an array of
    the same dataurl strings under `imageDataUrls`. Probe sweep against
    Seedream multi-ref / Nano Banana / Flux Kontext will confirm or correct
    per-model. Some NanoGPT model families may instead expect `images: [...]`
    or `reference_images: [...]`; capture in cassettes and override at the
    per-model level via a future preset family if the shape diverges.
    """
    return {"json": {"imageDataUrls": [to_dataurl(img) for img in images]}}


def openai_images_transform(images: list[bytes]) -> dict:
    """OpenAI `/v1/images/edits` multi-image multipart per gpt-image-1.

    The standard HTML-form-encoding for repeated fields: each image is sent as
    a separate `image[]` part. httpx accepts the list-of-tuples form of
    `files=` to produce this on the wire. Format is forced to PNG to match
    the existing single-image edit path's `image/png` content type; OpenAI
    accepts PNG and JPEG for edits but PNG is the safe default for the format
    we already detect upstream.
    """
    return {"files": [
        ("image[]", (f"input_{i}.{detect_image_format(img)}", img, f"image/{detect_image_format(img)}"))
        for i, img in enumerate(images)
    ]}


def openrouter_images_transform(images: list[bytes]) -> dict:
    """OpenRouter chat-completions multi-image content parts.

    Each reference becomes its own `image_url` content part appended after
    the text prompt. The dispatch path (`generate_image_via_chat`) prepends
    the text part, so the transform only emits the image parts. Format
    detected per-image for the dataurl MIME hint.
    """
    return {"content": [
        {"type": "image_url", "image_url": {"url": to_dataurl(img)}}
        for img in images
    ]}


# AIHubMix passes through to OpenAI for gpt-image-1, so the same multipart
# image[] shape works there. Other AIHubMix-hosted models (Wavespeed, Pruna,
# etc.) have varying shapes that need per-preset-family handling once those
# families exist; for now, AIHubMix multi-image reuses the OpenAI transform
# and degrades to first-image-only for non-OpenAI-shaped models.
aihubmix_images_transform = openai_images_transform


PRESETS: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api",
        "image_via": "chat",
        "video_via": "videos",
        "model_list": ["/v1/models"],
        "model_list_params": {"output_modalities": "all"},
        "auth_header": "Bearer",
        "extra_headers": {"HTTP-Referer": "http://localhost", "X-Title": "SD.Next"},
        "param_maps": {
            "image": {
                "prompt": ("_message_content", None),
                "n": ("n", lambda v: max(1, min(10, v))),
                "seed": ("seed", lambda v: v if v >= 0 else None),
            },
            "image_size_transform": size_transform_wxh,
            "images_transform": openrouter_images_transform,
            "chat": DEFAULT_CHAT_PARAMS,
            "tts": DEFAULT_TTS_PARAMS,
            "video": DEFAULT_VIDEO_PARAMS,
        },
        "input_limits": {
            "max_image_bytes": 20_000_000,
            "max_longest_side": 2048,
            "formats": ["webp", "jpeg", "png"],
            "transport": "base64",
        },
        "timeouts": {
            "cloud_image": 120,
            "cloud_chat": 120,
            "cloud_tts": 30,
            "cloud_stt": 60,
            "cloud_video": 900,
        },
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "image_via": "images",
        "video_via": "videos",
        "model_list": ["/v1/models"],
        "model_list_params": {},
        "auth_header": "Bearer",
        "extra_headers": {},
        "param_maps": {
            "image": DEFAULT_IMAGE_PARAMS,
            "image_size_transform": size_transform_wxh,
            "images_transform": openai_images_transform,
            "chat": DEFAULT_CHAT_PARAMS,
            "tts": DEFAULT_TTS_PARAMS,
            "video": DEFAULT_VIDEO_PARAMS,
        },
        "input_limits": {
            "max_image_bytes": 50_000_000,
            "max_longest_side": None,
            "formats": ["png", "jpeg", "webp"],
            "transport": "multipart",
        },
        "input_limits_overrides": {
            "dall-e-2": {
                "max_image_bytes": 4_000_000,
                "max_longest_side": 1024,
                "formats": ["png"],
            },
        },
        "timeouts": {
            "cloud_image": 120,
            "cloud_chat": 120,
            "cloud_tts": 30,
            "cloud_stt": 60,
            "cloud_video": 600,
        },
    },
    "nanogpt": {
        "base_url": "https://nano-gpt.com/api",
        "image_via": "dataurl",
        "video_via": "nanogpt",
        "model_list": ["/v1/models", "/v1/image-models", "/v1/video-models", "/v1/audio-models"],
        "model_list_params": {},
        "auth_header": "Bearer",
        "extra_headers": {},
        "param_maps": {
            "image": DEFAULT_IMAGE_PARAMS,
            "image_size_transform": size_transform_wxh,
            "images_transform": nanogpt_images_transform,
            "chat": DEFAULT_CHAT_PARAMS,
            "tts": DEFAULT_TTS_PARAMS,
            "video": DEFAULT_VIDEO_PARAMS,
        },
        "input_limits": {
            "max_image_bytes": 4_000_000,
            "max_longest_side": None,
            "formats": ["webp", "jpeg", "png"],
            "transport": "multipart",
        },
        "timeouts": {
            "cloud_image": 120,
            "cloud_chat": 120,
            "cloud_tts": 30,
            "cloud_stt": 60,
            "cloud_video": 600,
        },
    },
    "aihubmix": {
        "base_url": "https://aihubmix.com",
        "image_via": "images",
        "video_via": "videos",
        "model_list": ["/v1/models"],
        "model_list_params": {},
        "auth_header": "Bearer",
        "extra_headers": {},
        "param_maps": {
            "image": DEFAULT_IMAGE_PARAMS,
            "image_size_transform": size_transform_wxh,
            "images_transform": aihubmix_images_transform,
            "chat": DEFAULT_CHAT_PARAMS,
            "tts": DEFAULT_TTS_PARAMS,
            "video": DEFAULT_VIDEO_PARAMS,
        },
        "input_limits": {
            "max_image_bytes": 50_000_000,
            "max_longest_side": None,
            "formats": ["png", "jpeg", "webp"],
            "transport": "multipart",
        },
        "timeouts": {
            "cloud_image": 120,
            "cloud_chat": 120,
            "cloud_tts": 30,
            "cloud_stt": 60,
            "cloud_video": 600,
        },
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "image_via": "images",
        "video_via": "videos",
        "model_list": ["/v1/models"],
        "model_list_params": {},
        "auth_header": None,
        "extra_headers": {},
        "vision_input": "base64_only",
        "param_maps": {
            "image": {
                "prompt": ("prompt", None),
                "seed": ("seed", lambda v: v if v >= 0 else None),
            },
            "image_size_transform": size_transform_wxh,
            "chat": {
                "prompt": ("prompt", None),
                "temperature": ("temperature", lambda v: max(0.0, min(2.0, v))),
                "top_p": ("top_p", lambda v: max(0.0, min(1.0, v))),
                "seed": ("seed", lambda v: v if v >= 0 else None),
                "stop": ("stop", None),
            },
            "tts": DEFAULT_TTS_PARAMS,
            "video": DEFAULT_VIDEO_PARAMS,
        },
        "input_limits": {
            "max_image_bytes": 25_000_000,
            "max_longest_side": 1120,
            "formats": ["jpeg", "png", "webp"],
            "transport": "base64",
        },
        "timeouts": {
            "cloud_image": 300,
            "cloud_chat": 300,
            "cloud_tts": 60,
            "cloud_stt": 120,
            "cloud_video": 600,
        },
    },
    "custom": {
        "base_url": "",
        "image_via": "probe",
        "video_via": "probe",
        "model_list": ["/v1/models"],
        "model_list_params": {},
        "auth_header": "Bearer",
        "extra_headers": {},
        "param_maps": {
            "image": DEFAULT_IMAGE_PARAMS,
            "image_size_transform": size_transform_wxh,
            "chat": DEFAULT_CHAT_PARAMS,
            "tts": DEFAULT_TTS_PARAMS,
            "video": DEFAULT_VIDEO_PARAMS,
        },
        "input_limits": {
            "max_image_bytes": 20_000_000,
            "max_longest_side": None,
            "formats": ["webp", "jpeg", "png"],
            "transport": "multipart",
        },
        "timeouts": {
            "cloud_image": 120,
            "cloud_chat": 120,
            "cloud_tts": 30,
            "cloud_stt": 60,
            "cloud_video": 600,
        },
    },
}


def resolve_input_limits(preset: dict, model_id: str) -> dict:
    """Merge a preset's input_limits with model-specific overrides."""
    base = preset.get("input_limits", {})
    overrides = preset.get("input_limits_overrides", {})
    if not overrides:
        return base
    bare_id = model_id.split("/")[-1] if "/" in model_id else model_id
    for family, override in overrides.items():
        if bare_id == family or bare_id.startswith(f"{family}:"):
            return {**base, **override}
    return base
