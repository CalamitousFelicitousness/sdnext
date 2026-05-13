"""Cloud module package init.

On first import:
  1. Hydrates per-provider key OptionInfos for every provider already in
     shared.opts.cloud_providers.
  2. Registers the migrate_legacy_providers callback via on_app_started.

Public symbols are re-exported here for convenience, but consumers should
still import from the canonical sub-modules to keep import graphs explicit.
"""

from modules import script_callbacks

from modules.cloud import registry
from modules.cloud.errors import (
    AuthError,
    CloudError,
    ContentFilterError,
    InputValidationError,
    ModelNotFoundError,
    ProviderError,
    QuotaError,
    RateLimitError,
)
from modules.cloud.protocol import (
    AudioResult,
    ChatResult,
    CloudUsage,
    ImageResult,
    ProgressCallback,
    ProviderAdapter,
    TranscribeResult,
    VideoResult,
)
from modules.cloud.registry import (
    ProviderConfig,
    add_provider,
    get_adapter,
    get_provider,
    list_providers,
    refresh_models,
    remove_provider,
    update_provider,
    validate_provider,
)
from modules.cloud.text import caption, enhance_prompt, vqa
from modules.cloud.image import CloudImageGenResult, generate_image
from modules.cloud.migrate import migrate_legacy_providers


__all__ = [
    "AuthError",
    "CloudError",
    "ContentFilterError",
    "InputValidationError",
    "ModelNotFoundError",
    "ProviderError",
    "QuotaError",
    "RateLimitError",
    "AudioResult",
    "ChatResult",
    "CloudUsage",
    "ImageResult",
    "ProgressCallback",
    "ProviderAdapter",
    "TranscribeResult",
    "VideoResult",
    "ProviderConfig",
    "add_provider",
    "get_adapter",
    "get_provider",
    "list_providers",
    "refresh_models",
    "remove_provider",
    "update_provider",
    "validate_provider",
    "caption",
    "enhance_prompt",
    "vqa",
    "generate_image",
    "CloudImageGenResult",
]


# Hydrate per-provider key OptionInfos for any providers already configured.
# Must run before any cloud opts.set call so the suffix-based secrets routing
# in modules/options_handler.py:85-88 works.
registry.hydrate_key_options()

# Register the legacy migration. Fires once after FastAPI startup; idempotent.
script_callbacks.on_app_started(migrate_legacy_providers)
