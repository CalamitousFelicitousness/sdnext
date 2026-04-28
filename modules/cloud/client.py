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


def image_to_base64(img: 'PILImage.Image', *, max_dim: int = 2048, image_format: str = 'JPEG') -> str:
    work = img
    if max_dim and (img.width > max_dim or img.height > max_dim):
        work = img.copy()
        work.thumbnail((max_dim, max_dim))
    if image_format == 'JPEG' and work.mode != 'RGB':
        work = work.convert('RGB')
    buf = io.BytesIO()
    work.save(buf, format=image_format)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def image_to_data_url(img: 'PILImage.Image', *, max_dim: int = 2048, mime: str = 'image/jpeg') -> str:
    image_format = 'JPEG' if mime == 'image/jpeg' else 'PNG'
    b64 = image_to_base64(img, max_dim=max_dim, image_format=image_format)
    return f'data:{mime};base64,{b64}'
