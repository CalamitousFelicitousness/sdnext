"""Shared image-encoding helpers used by both `adapter.py` and `presets.py`.

These live outside `adapter.py` so that preset `images_transform` lambdas
(defined in `presets.py`) can call them without creating a circular import:
`adapter.py` already imports from `presets.py` via `resolve_input_limits`.
"""

import base64


def detect_image_format(image_data: bytes) -> str:
    """Sniff PNG / JPEG / WEBP from a magic-bytes header. Defaults to png.

    Same detector used by `OpenAICompatAdapter.detect_format`; lifted here so
    multi-image transform lambdas can call it without an adapter instance.
    """
    if image_data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if image_data[:2] == b"\xff\xd8":
        return "jpeg"
    if len(image_data) >= 12 and image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
        return "webp"
    return "png"


def to_dataurl(image_data: bytes) -> str:
    """Encode raw image bytes as a `data:image/<fmt>;base64,...` URL.

    Companion to `detect_image_format`; used by preset `images_transform`
    lambdas so each one doesn't re-implement base64 + format detection.
    """
    fmt = detect_image_format(image_data)
    b64 = base64.b64encode(image_data).decode("ascii")
    return f"data:image/{fmt};base64,{b64}"
