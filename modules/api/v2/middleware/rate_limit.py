"""
Rate limiting middleware for API v2
"""

import time
from collections import defaultdict, deque
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    Uses sliding window algorithm for rate limiting.
    """

    def __init__(self, app, requests: int = 100, period: int = 60):
        super().__init__(app)
        self.requests = requests
        self.period = period
        self.clients = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Get current time
        now = time.time()

        # Get client's request history
        client_requests = self.clients[client_ip]

        # Remove old requests outside the window
        while client_requests and client_requests[0] < now - self.period:
            client_requests.popleft()

        # Check rate limit
        if len(client_requests) >= self.requests:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": self.period},
                headers={"Retry-After": str(self.period), "X-RateLimit-Limit": str(self.requests), "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(now + self.period))},
            )

        # Add current request to history
        client_requests.append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests)
        response.headers["X-RateLimit-Remaining"] = str(self.requests - len(client_requests))
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))

        return response
