"""Async HTTP helpers for cloud providers.

Phase 1 providers use sync `requests` via modules.cloud.client. Phase 2 image/
video providers (REST-based — Civitai, BFL, NanoGPT in 2.1+) use httpx async
helpers from this module. The Google adapters use asyncio.to_thread to wrap the
sync google-genai SDK and do not need this module directly.

Reuse modules.cloud.client.mask_key / image_to_base64 / image_to_data_url
directly — those helpers are pure-sync and have no async equivalent.
"""
from __future__ import annotations
from typing import Optional
import httpx
from modules.logger import log


DEFAULT_TIMEOUT = 60.0
DEFAULT_CONNECT_TIMEOUT = 10.0


def make_client(timeout: float = DEFAULT_TIMEOUT, *, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
        follow_redirects=True,
        http2=False,
    )


async def post_json_async(client: httpx.AsyncClient, url: str, headers: dict, body: dict,
                          *, retries: int = 2) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                text = resp.text[:500] if resp.text else ''
                raise RuntimeError(f"HTTP {resp.status_code}: {text}")
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                log.debug(f'Cloud: post_json_async retry attempt={attempt + 1} url={url} error={e}')
            else:
                log.error(f'Cloud: post_json_async failed url={url} error={e}')
    raise last_exc if last_exc else RuntimeError("post_json_async failed")


async def download_async(client: httpx.AsyncClient, url: str, headers: Optional[dict] = None,
                         *, timeout: float = 120.0) -> bytes:
    resp = await client.get(url, headers=headers or {}, timeout=timeout)
    if resp.status_code >= 400:
        text = resp.text[:500] if resp.text else ''
        raise RuntimeError(f"HTTP {resp.status_code}: {text}")
    return resp.content
