"""
System information and control endpoints for API v2
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import platform
import psutil

router = APIRouter()


@router.get("/info")
async def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "api_version": "2.0.0"
    }


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
