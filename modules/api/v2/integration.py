"""
Integration module for ApiV2 with SDNext main application

This module provides the integration point for the v2 API
to be loaded alongside the existing v1 API.
"""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI

from .api import ApiV2


# Global instance
_api_v2_instance: Optional[ApiV2] = None


def initialize_api_v2(app: FastAPI, queue_lock: Optional[asyncio.Lock] = None) -> ApiV2:
    """
    Initialize and register the v2 API with the main application.
    
    This function should be called from the main SDNext initialization
    to set up the v2 API alongside v1.
    
    Args:
        app: The main FastAPI application instance
        queue_lock: Optional async lock for queue management
        
    Returns:
        The initialized ApiV2 instance
    """
    global _api_v2_instance
    
    if _api_v2_instance is not None:
        logging.warning("ApiV2 already initialized")
        return _api_v2_instance
    
    # Create ApiV2 instance
    _api_v2_instance = ApiV2(app=app, queue_lock=queue_lock)
    
    # Register routes and middleware
    _api_v2_instance.register()
    
    logging.info("SDNext v2 API initialized and registered")
    
    return _api_v2_instance


def get_api_v2() -> Optional[ApiV2]:
    """
    Get the global ApiV2 instance.
    
    Returns:
        The ApiV2 instance or None if not initialized
    """
    return _api_v2_instance


# Convenience functions for external use

async def broadcast_v2_update(update: dict):
    """Broadcast an update through v2 WebSockets."""
    if _api_v2_instance:
        await _api_v2_instance.broadcast_update(update)
