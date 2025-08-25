"""
Authentication middleware for API v2
"""

from typing import Optional, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import jwt
import time


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT authentication middleware.

    Validates JWT tokens and adds user information to request state.
    """

    def __init__(self, app, secret: str, algorithm: str = "HS256", exempt_paths: Optional[list] = None):
        super().__init__(app)
        self.secret = secret
        self.algorithm = algorithm
        self.exempt_paths = exempt_paths or ["/sdapi/v2/docs", "/sdapi/v2/redoc", "/sdapi/v2/openapi.json", "/sdapi/v2/system/health"]

    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Missing or invalid authorization header"})

        token = auth_header.split(" ")[1]

        try:
            # Verify and decode token
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])

            # Check expiration
            if payload.get("exp") and payload["exp"] < time.time():
                return JSONResponse(status_code=401, content={"error": "Token expired"})

            # Add user info to request state
            request.state.user = payload
            request.state.authenticated = True

        except jwt.InvalidTokenError as e:
            return JSONResponse(status_code=401, content={"error": f"Invalid token: {str(e)}"})

        # Process request
        response = await call_next(request)
        return response
