"""Canary probe for OpenRouter image-gen models.

Verifies whether the `size` parameter is consumed, silently ignored, or
rejected across the five OpenRouter sub-patterns observed in discovery:

  1. Flux variant (chat-style supported_params): black-forest-labs/flux.2-pro
  2. Seedream (chat-style supported_params):     bytedance-seed/seedream-4.5
  3. Gemini (chat-style supported_params):       google/gemini-3-pro-image-preview
  4. GPT-image (chat-style supported_params):    openai/gpt-5-image
  5. Recraft (no supported_params):              recraft/recraft-v3

Each probe sends a minimal chat-completions request with `size="1024x1024"`
plus `modalities=["image", "text"]` (OpenAI's documented way to request image
output). Captures full response status / body for human review.

Output written to test/cloud/discovery/openrouter/canary_results.json. Does
NOT modify size_constraints.json directly; the human reviews canary output
and decides whether to codify all 28 OpenRouter image models as null, or
pursue per-family probes.
"""

import sys
from pathlib import Path

ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time

from modules.cloud import registry
from modules.cloud.errors import CloudError
from modules.logger import log


CANARY_MODELS = [
    "black-forest-labs/flux.2-pro",
    "bytedance-seed/seedream-4.5",
    "google/gemini-3-pro-image-preview",
    "openai/gpt-5-image",
    "recraft/recraft-v3",
]
CANARY_SIZE = "1024x1024"
CANARY_PROMPT = "a small red square on a white background, simple shape"
OUTPUT_PATH = REPO_ROOT / "test" / "cloud" / "discovery" / "openrouter" / "canary_results.json"


def extract_image_dimensions(response: dict) -> tuple[int, int] | None:
    """Try to decode the returned image and read its (width, height).

    Handles three shapes seen in OpenRouter chat-completions image responses:
      - choices[0].message.images[] with {"image_url": {"url": "data:image/...base64,..."}}
      - choices[0].message.content as base64 data URL string
      - choices[0].message.content as a URL we have to fetch

    Returns None if no decodable image is found.
    """
    import base64
    import io
    try:
        from PIL import Image
    except ImportError:
        return None
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    candidates: list[str] = []
    for img in (message.get("images") or []):
        url = (img.get("image_url") or {}).get("url") or img.get("url")
        if url:
            candidates.append(url)
    content = message.get("content")
    if isinstance(content, str) and content:
        candidates.append(content)
    for url_or_data in candidates:
        if not url_or_data.startswith("data:"):
            continue  # URL fetching deferred; data: URIs handle inline
        try:
            payload = url_or_data.split(",", 1)[1]
            img_bytes = base64.b64decode(payload)
            with Image.open(io.BytesIO(img_bytes)) as pil:
                return (pil.width, pil.height)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    return None


def canary_probe(adapter, model_id: str) -> dict:
    """Send one chat-completions request with size and modalities=['image'];
    capture full response and decoded image dimensions if extractable."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": CANARY_PROMPT}],
        "modalities": ["image"],
        "size": CANARY_SIZE,
        "max_tokens": 4096,
    }
    started = time.time()
    try:
        response = adapter.transport.post("/v1/chat/completions", json=body)
        elapsed = time.time() - started
        dims = extract_image_dimensions(response)
        size_honored = None
        if dims:
            requested_w, requested_h = (int(s) for s in CANARY_SIZE.split("x"))
            size_honored = dims == (requested_w, requested_h)
        usage = response.get("usage") or {}
        choices = response.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        return {
            "model_id": model_id,
            "request_body": body,
            "outcome": "ok",
            "status_class": "2xx",
            "elapsed_seconds": round(elapsed, 2),
            "returned_dims": list(dims) if dims else None,
            "size_honored": size_honored,
            "finish_reason": finish_reason,
            "usage": usage,
            "full_response": response,
        }
    except CloudError as e:
        elapsed = time.time() - started
        return {
            "model_id": model_id,
            "request_body": body,
            "outcome": "error",
            "status_class": type(e).__name__,
            "elapsed_seconds": round(elapsed, 2),
            "error_message": str(e),
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        elapsed = time.time() - started
        return {
            "model_id": model_id,
            "request_body": body,
            "outcome": "error",
            "status_class": type(e).__name__,
            "elapsed_seconds": round(elapsed, 2),
            "error_message": str(e),
        }


def main() -> int:
    adapter = registry.get_adapter("openrouter")
    results: list[dict] = []
    log.info(f"Cloud canary: starting OpenRouter probe across {len(CANARY_MODELS)} models")
    for model_id in CANARY_MODELS:
        log.info(f"Cloud canary: probing {model_id}")
        result = canary_probe(adapter, model_id)
        results.append(result)
        log.info(f"Cloud canary: {model_id} outcome={result['outcome']} status_class={result['status_class']} elapsed={result['elapsed_seconds']}s")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    log.info(f"Cloud canary: results saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
