from __future__ import annotations
import io
import json
import base64
from typing import Callable, Iterator, Optional, TYPE_CHECKING
import requests
from modules.logger import log


if TYPE_CHECKING:
    from PIL import Image as PILImage


def mask_key(k: Optional[str]) -> str:
    if not k:
        return '(none)'
    if len(k) <= 4:
        return '****'
    return '...' + k[-4:]


def post_json(url: str, headers: dict, body: dict, *, timeout: int = 60, retries: int = 2) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                log.debug(f'Cloud: post_json retry attempt={attempt + 1} url={url} error={e}')
            else:
                log.error(f'Cloud: post_json failed url={url} error={e}')
    raise last_exc if last_exc else RuntimeError("post_json failed")


def stream_sse(url: str, headers: dict, body: dict, *, timeout: int = 180,
               event_extractor: Optional[Callable[[dict], Optional[str]]] = None) -> Iterator[str]:
    extractor = event_extractor if event_extractor is not None else (
        lambda evt: evt.get('delta') if isinstance(evt, dict) else None
    )
    with requests.post(url, headers=headers, json=body, stream=True, timeout=timeout) as resp:
        if resp.status_code >= 400:
            text = resp.text[:500] if hasattr(resp, 'text') else ''
            raise RuntimeError(f"HTTP {resp.status_code}: {text}")
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith('data:'):
                continue
            payload = line[len('data:'):].strip()
            if payload == '[DONE]':
                return
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunk = extractor(evt)
            if chunk:
                yield chunk


def encode_image(img: 'PILImage.Image', *, max_dim: int = 2048, image_format: str = 'JPEG', quality: int = 85) -> tuple[bytes, 'PILImage.Image']:
    """Encode a PIL image to bytes, thumbnailing to max_dim and converting mode as needed.

    Returns (raw_bytes, working_copy) so callers can inspect size before base64-encoding.
    """
    work = img
    if max_dim and (img.width > max_dim or img.height > max_dim):
        work = img.copy()
        work.thumbnail((max_dim, max_dim))
    if image_format == 'JPEG' and work.mode != 'RGB':
        work = work.convert('RGB')
    buf = io.BytesIO()
    save_kwargs: dict = {'format': image_format}
    if image_format == 'JPEG':
        save_kwargs['quality'] = quality
    work.save(buf, **save_kwargs)
    return buf.getvalue(), work


def image_to_base64(img: 'PILImage.Image', *, max_dim: int = 2048, image_format: str = 'JPEG',
                    max_bytes: int = 0, quality: int = 85) -> str:
    """Encode a PIL image to a base64 string that fits within max_bytes.

    Reduction strategy when the result exceeds max_bytes:
      1. Lower JPEG quality in steps (85 → 70 → 55 → 40).
      2. If still too large, halve dimensions and retry from quality 85.
    PNG images are re-encoded as JPEG before quality reduction since PNG
    quality is not tunable.

    Args:
        max_bytes: budget for the base64-encoded string (0 = no limit).
                   Provider limits are on the encoded form — NanoGPT 4 MB,
                   Anthropic 5 MB, OpenAI 20 MB. Pass the provider's limit
                   directly; this function checks len(b64_string) against it.
    """
    quality_steps = (quality, 70, 55, 40)
    dim = max_dim
    for _ in range(4):
        for q in quality_steps:
            fmt = 'JPEG' if (image_format == 'JPEG' or (max_bytes and q < quality)) else image_format
            raw, _ = encode_image(img, max_dim=dim, image_format=fmt, quality=q)
            b64 = base64.b64encode(raw).decode('ascii')
            if max_bytes <= 0 or len(b64) <= max_bytes:
                return b64
        dim = max(dim // 2, 256)
    raw, _ = encode_image(img, max_dim=256, image_format='JPEG', quality=40)
    return base64.b64encode(raw).decode('ascii')


def image_to_data_url(img: 'PILImage.Image', *, max_dim: int = 2048, mime: str = 'image/jpeg',
                      max_bytes: int = 0) -> str:
    """Encode a PIL image as a data URL, optionally clamped to max_bytes.

    The max_bytes budget applies to the full data URL string (prefix + base64).
    """
    image_format = 'JPEG' if mime == 'image/jpeg' else 'PNG'
    prefix = f'data:{mime};base64,'
    budget = max(0, max_bytes - len(prefix)) if max_bytes > 0 else 0
    b64 = image_to_base64(img, max_dim=max_dim, image_format=image_format, max_bytes=budget)
    return f'{prefix}{b64}'


def parse_resolution(res: str) -> 'tuple[int, int] | None':
    """Parse a resolution string into (width, height) pixels.

    Handles five format families:
      WxH (``1024x1024``), W*H (``1024*1024``), W:H ratio (``16:9``),
      K-suffix (``2k``), p-suffix (``1080p``), bare digit (``1024``).
    Returns None for tokens that carry no pixel information (``auto``, ``default``).
    """
    if not res or not isinstance(res, str):
        return None
    for sep in ('x', '*'):
        if sep in res:
            parts = res.split(sep, 1)
            if parts[0].isdigit() and parts[1].isdigit():
                return (int(parts[0]), int(parts[1]))
    if ':' in res:
        parts = res.split(':', 1)
        if parts[0].isdigit() and parts[1].isdigit():
            w, h = int(parts[0]), int(parts[1])
            scale = 1024 / max(w, h, 1)
            return (int(w * scale), int(h * scale))
    lower = res.lower().strip()
    k_map = {'1k': 1024, '2k': 2048, '4k': 4096, '8k': 8192}
    if lower in k_map:
        dim = k_map[lower]
        return (dim, dim)
    p_map = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (854, 480)}
    if lower in p_map:
        return p_map[lower]
    if res.isdigit():
        return (int(res), int(res))
    return None


def closest_resolution(width: int, height: int, resolutions: list[str]) -> str:
    """Pick the resolution string from *resolutions* closest to (width, height).

    Comparison is done in pixel space after parsing each candidate. Aspect ratio
    proximity is weighted 5x over pixel-count proximity so the mapper favours
    the right shape over the right size. Returns the provider's original token.

    Fallbacks: empty list → ``'{width}x{height}'``; all-unparseable → ``'auto'``
    if present, else first item.
    """
    if not resolutions:
        return f'{width}x{height}'
    candidates: list[tuple[str, tuple[int, int]]] = []
    for res in resolutions:
        dims = parse_resolution(res)
        if dims:
            candidates.append((res, dims))
    if not candidates:
        if 'auto' in resolutions:
            return 'auto'
        return resolutions[0]
    target_ar = (width or 1) / max(height or 1, 1)
    target_px = (width or 1024) * (height or 1024)

    def score(entry):
        cw, ch = entry[1]
        ar = cw / max(ch, 1)
        px = cw * ch
        ar_diff = abs(ar - target_ar) / max(target_ar, 0.01)
        px_ratio = max(px, target_px) / max(min(px, target_px), 1) - 1
        return ar_diff * 5 + px_ratio

    return min(candidates, key=score)[0]
