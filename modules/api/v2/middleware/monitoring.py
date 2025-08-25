"""
Monitoring middleware for API v2
"""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring API requests and collecting metrics.
    """

    def __init__(self, app, metrics: dict):
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Update metrics
        self.metrics["requests_total"] += 1
        self.metrics[f"requests_{response.status_code}"] = self.metrics.get(f"requests_{response.status_code}", 0) + 1
        self.metrics["total_process_time"] = self.metrics.get("total_process_time", 0) + process_time

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response
