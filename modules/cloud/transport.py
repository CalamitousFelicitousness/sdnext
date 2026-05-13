"""Sync HTTP transport for cloud adapters.

httpx.Client wrapper with retries, rate-limit tracking, and TTL caching.
Composed into adapter.OpenAICompatAdapter.

Sync (httpx.Client, not AsyncClient) to match the rest of the cloud module.
Retry sleeps are interruptible via shared.state.interrupted so a
user-pressed cancel during a backoff window aborts cleanly.
"""

import contextlib
import os
import time

import httpx
from modules.logger import log
from modules import shared

from modules.cloud.errors import (
    AuthError,
    CloudError,
    ContentFilterError,
    ModelNotFoundError,
    ProviderError,
    QuotaError,
    RateLimitError,
)


# SD_CLOUD_DEBUG=1 enables verbose per-request logging. Mirrors gallery.py:18 pattern.
debug = log.debug if os.environ.get("SD_CLOUD_DEBUG") else lambda *args, **kwargs: None


class HttpTransport:
    """httpx.Client with provider-aware retries, error mapping, and caching."""

    def __init__(self, provider_id: str, base_url: str, preset: dict, key: str):
        self.provider_id = provider_id
        self.preset = preset
        self.client = httpx.Client(
            base_url=self.normalize_base_url(base_url),
            headers=self.build_headers(key),
            timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
        )
        self.cache: dict[str, tuple[float, object]] = {}
        self.rate_limit_remaining: int | None = None
        self.rate_limit_reset: float | None = None

    def normalize_base_url(self, base_url: str) -> str:
        # Every preset path begins with /v1/ (e.g. /v1/chat/completions). httpx
        # concatenates base_url + path rather than urljoin'ing, so a base_url
        # ending in /v1 produces /v1/v1/... requests. The OpenAI Python SDK
        # convention is base_url WITH /v1, which users naturally carry over
        # when configuring a custom provider. Strip it here so both work.
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            stripped = normalized[:-3]
            log.info(f"Cloud: base_url normalized provider={self.provider_id} from={base_url!r} to={stripped!r}")
            return stripped
        return normalized

    def build_headers(self, key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        auth_type = self.preset.get("auth_header")
        if auth_type and key:
            headers["Authorization"] = f"{auth_type} {key}"
        for k, v in self.preset.get("extra_headers", {}).items():
            if v:
                headers[k] = v
        return headers

    def get(self, path: str, params: dict | None = None) -> dict:
        return self.request("GET", path, params=params)

    def post(self, path: str, json: dict | None = None, **kw) -> dict:
        return self.request("POST", path, json=json, **kw)

    def get_cached(self, path: str, ttl: int = 300, params: dict | None = None) -> object:
        cache_key = f"{path}?{params}" if params else path
        now = time.time()
        if cache_key in self.cache:
            expires, data = self.cache[cache_key]
            if now < expires:
                return data
        result = self.get(path, params=params)
        self.cache[cache_key] = (now + ttl, result)
        return result

    def invalidate_cache(self, path: str | None = None) -> None:
        if path is None:
            self.cache.clear()
        else:
            self.cache = {k: v for k, v in self.cache.items() if not k.startswith(path)}

    def request(self, method: str, path: str, **kw) -> dict:
        max_attempts = 3
        backoff = 1.0
        t0 = time.time()
        provider_id = self.provider_id

        for attempt in range(max_attempts):
            if shared.state.interrupted:
                raise ProviderError("Interrupted by user before request", provider=provider_id)
            try:
                response = self.client.request(method, path, **kw)
                self.update_rate_limits(response.headers)
                elapsed = time.time() - t0

                if response.status_code == 429:
                    retry_after = self.parse_retry_after(response.headers)
                    if attempt < 1 and retry_after and retry_after < 60:
                        log.warning(f"Cloud: rate-limited provider={provider_id} {method} {path} retry_after={retry_after:.1f}s attempt={attempt + 1}")
                        if not self.sleep_interruptible(retry_after):
                            raise ProviderError("Interrupted by user during rate-limit backoff", provider=provider_id)
                        continue
                    log.warning(f"Cloud: rate-limited provider={provider_id} {method} {path} retry_after={retry_after} (giving up)")
                    raise RateLimitError(self.extract_error_message(response), provider=provider_id, retry_after=retry_after)

                if response.status_code in (500, 502, 503) and attempt < max_attempts - 1:
                    delay = backoff * (2**attempt)
                    log.warning(f"Cloud: server error provider={provider_id} {method} {path} status={response.status_code} retry_in={delay:.1f}s attempt={attempt + 1}/{max_attempts}")
                    if not self.sleep_interruptible(delay):
                        raise ProviderError("Interrupted by user during server-error backoff", provider=provider_id)
                    continue

                if response.status_code >= 400:
                    log.warning(f"Cloud: request failed provider={provider_id} {method} {path} status={response.status_code} time={elapsed:.2f}s msg={self.extract_error_message(response)!r}")
                    self.raise_for_status(response)

                debug(f"Cloud: request ok provider={provider_id} {method} {path} status={response.status_code} time={elapsed:.2f}s rl_remaining={self.rate_limit_remaining}")
                return response.json()

            except httpx.TimeoutException as e:
                if attempt < max_attempts - 1:
                    delay = backoff * (2**attempt)
                    log.warning(f"Cloud: timeout provider={provider_id} {method} {path} retry_in={delay:.1f}s attempt={attempt + 1}/{max_attempts}")
                    if not self.sleep_interruptible(delay):
                        raise ProviderError("Interrupted by user during timeout backoff", provider=provider_id) from e
                    continue
                log.error(f"Cloud: timeout provider={provider_id} {method} {path} time={time.time() - t0:.2f}s (giving up)")
                raise ProviderError("Request timed out", provider=provider_id) from e

            except httpx.ConnectError as e:
                if attempt < max_attempts - 1:
                    delay = backoff * (2**attempt)
                    log.warning(f"Cloud: connect error provider={provider_id} {method} {path} retry_in={delay:.1f}s attempt={attempt + 1}/{max_attempts}: {e}")
                    if not self.sleep_interruptible(delay):
                        raise ProviderError("Interrupted by user during connect backoff", provider=provider_id) from e
                    continue
                log.error(f"Cloud: connect failed provider={provider_id} {method} {path}: {e}")
                raise ProviderError("Connection failed", provider=provider_id) from e

            except CloudError:
                raise

            except Exception as e:
                if attempt < max_attempts - 1:
                    delay = backoff * (2**attempt)
                    log.warning(f"Cloud: unexpected error provider={provider_id} {method} {path} retry_in={delay:.1f}s attempt={attempt + 1}/{max_attempts}: {e}")
                    if not self.sleep_interruptible(delay):
                        raise ProviderError("Interrupted by user during error backoff", provider=provider_id) from e
                    continue
                log.error(f"Cloud: unexpected error provider={provider_id} {method} {path}: {e}")
                raise ProviderError(str(e), provider=provider_id) from e

        raise ProviderError("Max retries exceeded", provider=provider_id)

    def sleep_interruptible(self, duration: float, tick: float = 0.5) -> bool:
        """Sleep up to duration seconds, polling shared.state.interrupted every tick.

        Returns True if completed normally, False if interrupted.
        """
        end = time.time() + duration
        while True:
            if shared.state.interrupted:
                return False
            remaining = end - time.time()
            if remaining <= 0:
                return True
            time.sleep(min(tick, remaining))

    def raise_for_status(self, response: httpx.Response) -> None:
        message = self.extract_error_message(response)
        status = response.status_code
        provider = self.provider_id
        if status == 401:
            raise AuthError(message, provider)
        if status == 402:
            raise QuotaError(message, provider)
        if status == 403:
            raise ContentFilterError(message, provider)
        if status == 404:
            raise ModelNotFoundError(message, provider)
        if status == 429:
            raise RateLimitError(message, provider, self.parse_retry_after(response.headers))
        raise ProviderError(message, provider, status=status)

    def extract_error_message(self, response: httpx.Response) -> str:
        try:
            data = response.json()
            error = data.get("error", {})
            if isinstance(error, dict):
                return error.get("message", "") or str(error)
            return str(error)
        except Exception:
            return response.text[:200] if response.text else f"HTTP {response.status_code}"

    def update_rate_limits(self, headers: httpx.Headers) -> None:
        remaining = headers.get("x-ratelimit-remaining") or headers.get("x-ratelimit-remaining-requests")
        if remaining is not None:
            with contextlib.suppress(ValueError):
                self.rate_limit_remaining = int(remaining)
        reset = headers.get("x-ratelimit-reset") or headers.get("x-ratelimit-reset-requests")
        if reset is not None:
            try:
                val = float(reset)
                self.rate_limit_reset = val if val > 1_000_000_000 else time.time() + val
            except ValueError:
                pass

    def parse_retry_after(self, headers: httpx.Headers) -> float | None:
        raw = headers.get("retry-after") or headers.get("x-ratelimit-reset")
        if raw is None:
            return None
        try:
            val = float(raw)
            if val > 1_000_000_000:
                return max(0.0, val / 1000 - time.time())
            return val
        except ValueError:
            return None

    def close(self) -> None:
        self.client.close()
