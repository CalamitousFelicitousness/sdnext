"""
Main ApiV2 class for SDNext v2 API

This class manages the FastAPI application setup, route registration,
and lifecycle management for the v2 API endpoints.
"""

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from modules.shared import log

# Import routers (to be implemented)
# from .routers import system, generation, canvas, models, queue
# from .websockets import ConnectionManager
# from .middleware import AuthMiddleware, RateLimitMiddleware


class ApiV2:
    """
    Main API v2 class that manages FastAPI application and routes.
    
    This class provides:
    - Async-first design with proper lifecycle management
    - WebSocket and SSE support for real-time updates
    - Enhanced queue management
    - Multi-tier caching
    - Proper dependency injection
    """
    
    def __init__(self, app: FastAPI, queue_lock: asyncio.Lock):
        """
        Initialize the ApiV2 instance.
        
        Args:
            app: The main FastAPI application instance
            queue_lock: Async lock for queue management
        """
        self.app = app
        self.queue_lock = queue_lock
        self.router = APIRouter(prefix="/sdapi/v2")
        # self.connection_manager = ConnectionManager()
        self.logger = logging.getLogger(__name__)
        
        # Store initialization state
        self._initialized = False
        
    async def startup(self):
        """
        Async startup handler for initializing resources.
        
        This method:
        - Initializes Redis connection for distributed queue
        - Sets up background task workers
        - Initializes cache managers
        - Prepares resource pools
        """
        if self._initialized:
            return
            
        self.logger.info("Starting SDNext v2 API...")
        
        # TODO: Initialize Redis connection
        # TODO: Set up background task workers
        # TODO: Initialize cache managers
        # TODO: Set up resource pools
        
        self._initialized = True
        self.logger.info("SDNext v2 API started successfully")
        
    async def shutdown(self):
        """
        Async shutdown handler for cleaning up resources.
        """
        self.logger.info("Shutting down SDNext v2 API...")
        
        # TODO: Close Redis connections
        # TODO: Cancel background tasks
        # TODO: Cleanup resource pools
        
        self._initialized = False
        self.logger.info("SDNext v2 API shutdown complete")
        
    def register(self):
        """
        Register all v2 API routes and middleware.
        
        This method:
        - Registers all domain routers
        - Sets up middleware stack
        - Configures CORS settings
        - Adds WebSocket endpoints
        """
        self.logger.info("Registering SDNext v2 API routes...")
        
        # TODO: Register domain routers
