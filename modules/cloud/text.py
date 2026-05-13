"""Cloud text functions: prompt enhancement, captioning, VQA.

Sync. Each function wraps adapter.chat() with shared.state lifecycle
(state.begin / state.end / state.textinfo) and surfaces upstream failures
as modules.cloud.errors.* types.

The Cloud-Text state title is shared across all three entry points so
state-history filtering and cost dashboards can group cloud text activity
uniformly.
"""

import base64

from modules import shared
from modules.logger import log

from modules.cloud import registry
from modules.cloud.errors import CloudError, ProviderError


STATE_TITLE = "Cloud-Text"

DEFAULT_ENHANCE_SYSTEM_NSFW_OK = (
    "You are a creative writing assistant. "
    "Enhance the user's prompt with vivid, evocative details while preserving its intent. "
    "NSFW content and nudity are allowed and should be described fully if present. "
    "Output a single concise paragraph without explanations or meta-commentary."
)
DEFAULT_ENHANCE_SYSTEM_NSFW_NO = (
    "You are a creative writing assistant. "
    "Enhance the user's prompt with vivid, evocative details while preserving its intent. "
    "NSFW content and nudity are not allowed. "
    "Output a single concise paragraph without explanations or meta-commentary."
)
DEFAULT_CAPTION_SYSTEM = (
    "You are an image captioning expert. Describe the image accurately and in detail."
)
DEFAULT_VQA_SYSTEM = (
    "You are a visual question answering expert. Answer questions about images concisely and accurately."
)


def detect_image_format(image_bytes: bytes) -> str:
    """Sniff PNG / JPEG / WEBP from the file header. Defaults to png."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if image_bytes[:2] == b"\xff\xd8":
        return "jpeg"
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "png"


def build_vision_messages(image_bytes: bytes, question: str, system_prompt: str) -> list[dict]:
    """Build OpenAI-style chat messages with a base64 image payload."""
    fmt = detect_image_format(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    user_content: list[dict] = [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}},
    ]
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def call_chat(provider_id: str, model: str, messages: list[dict]) -> str:
    """Drive adapter.chat() and return the content string. Raises CloudError on failure."""
    adapter = registry.get_adapter(provider_id)
    result = adapter.chat({"model": model, "messages": messages})
    content = result.content or ""
    if not content:
        raise ProviderError("Empty response from provider", provider=provider_id)
    return content.strip()


def enhance_prompt(
    prompt: str,
    provider_id: str,
    model: str,
    *,
    system_prompt: str = "",
    nsfw: bool = True,
) -> str:
    """Enhance a prompt via a cloud LLM."""
    if not prompt or not prompt.strip():
        raise ValueError("enhance_prompt: prompt is empty")
    if not system_prompt:
        system_prompt = DEFAULT_ENHANCE_SYSTEM_NSFW_OK if nsfw else DEFAULT_ENHANCE_SYSTEM_NSFW_NO
    jobid = shared.state.begin(STATE_TITLE, api=True)
    shared.state.textinfo = f"Cloud: {provider_id} / {model}"
    log.info(f"Cloud: enhance_prompt provider={provider_id} model={model} chars={len(prompt)} nsfw={nsfw}")
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()},
        ]
        return call_chat(provider_id, model, messages)
    except CloudError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise ProviderError(str(e), provider=provider_id) from e
    finally:
        shared.state.end(jobid)


def caption(
    image_bytes: bytes,
    provider_id: str,
    model: str,
    *,
    prompt: str = "Describe this image in detail.",
) -> str:
    """Caption an image via a cloud VLM."""
    if not image_bytes:
        raise ValueError("caption: image_bytes is empty")
    jobid = shared.state.begin(STATE_TITLE, api=True)
    shared.state.textinfo = f"Cloud: {provider_id} / {model}"
    log.info(f"Cloud: caption provider={provider_id} model={model} bytes={len(image_bytes)}")
    try:
        messages = build_vision_messages(image_bytes, prompt, DEFAULT_CAPTION_SYSTEM)
        return call_chat(provider_id, model, messages)
    except CloudError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise ProviderError(str(e), provider=provider_id) from e
    finally:
        shared.state.end(jobid)


def vqa(
    image_bytes: bytes,
    question: str,
    provider_id: str,
    model: str,
) -> str:
    """Answer a question about an image via a cloud VLM."""
    if not image_bytes:
        raise ValueError("vqa: image_bytes is empty")
    if not question or not question.strip():
        raise ValueError("vqa: question is empty")
    jobid = shared.state.begin(STATE_TITLE, api=True)
    shared.state.textinfo = f"Cloud: {provider_id} / {model}"
    log.info(f"Cloud: vqa provider={provider_id} model={model} bytes={len(image_bytes)} q_chars={len(question)}")
    try:
        messages = build_vision_messages(image_bytes, question.strip(), DEFAULT_VQA_SYSTEM)
        return call_chat(provider_id, model, messages)
    except CloudError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise ProviderError(str(e), provider=provider_id) from e
    finally:
        shared.state.end(jobid)
