"""
SDNext v2 API Package

A modern, async-first API implementation for SDNext that provides:
- WebSocket support for real-time updates
- Server-sent events for progress streaming
- Enhanced queue management with Redis support
- Multi-tier caching
- Improved performance and scalability

This package is designed to be UI-agnostic and does not depend on Gradio.
"""

from .api import ApiV2

__version__ = "2.0.0"
__all__ = [
    "ApiV2",
]

# Version compatibility check
import sys
if sys.version_info < (3, 10):
