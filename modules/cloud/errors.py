"""Cloud adapter error hierarchy.

Each subclass maps to an HTTP status from the upstream cloud provider.
text.py and api_v1.py catch by type to determine retry policy and the
error envelope returned to V1 callers.
"""


class CloudError(Exception):
    """Base for all cloud adapter errors."""

    def __init__(self, message: str, provider: str | None = None, status: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status = status


class AuthError(CloudError):
    """401: invalid or missing API key."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider, status=401)


class QuotaError(CloudError):
    """402: insufficient credits or balance exhausted."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider, status=402)


class ContentFilterError(CloudError):
    """403: content policy violation."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider, status=403)


class ModelNotFoundError(CloudError):
    """404: model id not recognised by the provider."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider, status=404)


class RateLimitError(CloudError):
    """429: rate limited. Check retry_after for backoff."""

    def __init__(self, message: str, provider: str | None = None, retry_after: float | None = None):
        super().__init__(message, provider, status=429)
        self.retry_after = retry_after


class ProviderError(CloudError):
    """5xx or transport failure: transient upstream issue, generally safe to retry."""

    def __init__(self, message: str, provider: str | None = None, status: int = 500):
        super().__init__(message, provider, status=status)


class InputValidationError(CloudError):
    """400: pre-upload input violates a constraint (size, dimensions, format).

    Raised by the adapter before sending to the provider, so the failure is
    attributable to caller input rather than to the provider. The api_v1
    handler maps this to HTTP 400 with kind="input_validation".
    """

    def __init__(self, message: str, provider: str | None = None,
                 field: str | None = None, limit: object = None):
        super().__init__(message, provider, status=400)
        self.field = field
        self.limit = limit
