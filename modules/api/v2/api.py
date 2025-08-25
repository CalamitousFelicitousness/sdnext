"""
Main ApiV2 class for SDNext v2 API

This class manages the FastAPI application setup, route registration,
and lifecycle management for the v2 API endpoints.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from contextlib import asynccontextmanager
from collections import defaultdict
import traceback

from fastapi import FastAPI, APIRouter, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Import internal modules
try:
    from modules.shared import log, opts, cmd_opts
    from modules import timer
except ImportError:
    # Fallback for testing
    log = logging.getLogger(__name__)
    opts = None
    cmd_opts = None
    timer = None

from .config import config
from .dependencies import get_queue_lock, get_redis_client
from .websockets.manager import ConnectionManager
from .middleware.auth import JWTAuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.monitoring import MonitoringMiddleware

# Import routers
from .routers.system import router as system_router
from .routers.generation import router as generation_router
from .routers.canvas import router as canvas_router
from .routers.models import router as models_router
from .routers.queue import router as queue_router
from .routers.events import router as events_router
from .routers.assets import router as assets_router


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests."""

    def __init__(self, app, logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        self.logger.debug(f"Request: {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        self.logger.debug(f"Response: {request.url.path} - {response.status_code} - {process_time:.3f}s")

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        return response


class ApiV2:
    """
    Main API v2 class that manages FastAPI application and routes.

    This class provides:
    - Async-first design with proper lifecycle management
    - WebSocket and SSE support for real-time updates
    - Enhanced queue management with Redis support
    - Multi-tier caching system
    - Proper dependency injection
    - Comprehensive error handling
    """

    def __init__(self, app: Optional[FastAPI] = None, queue_lock: Optional[asyncio.Lock] = None):
        """
        Initialize the ApiV2 instance.

        Args:
            app: The main FastAPI application instance (optional, will create if not provided)
            queue_lock: Async lock for queue management (optional, will create if not provided)
        """
        # Setup logging
        self.logger = log if log else logging.getLogger(__name__)
        self.logger.info("Initializing SDNext v2 API...")

        # Create or use provided FastAPI app
        if app is None:
            self.app = FastAPI(title="SDNext v2 API", description="Modern async API for SDNext", version="2.0.0", docs_url="/sdapi/v2/docs", redoc_url="/sdapi/v2/redoc", openapi_url="/sdapi/v2/openapi.json")
            self.owns_app = True
        else:
            self.app = app
            self.owns_app = False

        # Setup queue lock
        self.queue_lock = queue_lock if queue_lock else asyncio.Lock()

        # Create main router with v2 prefix
        self.router = APIRouter(prefix="/sdapi/v2")

        # Initialize connection manager for WebSockets
        self.connection_manager = ConnectionManager()

        # Track initialization state
        self._initialized = False
        self._startup_complete = False

        # Service instances (to be initialized in startup)
        self.redis_client = None
        self.cache_service = None
        self.queue_service = None
        self.storage_service = None

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        # Metrics
        self.metrics = defaultdict(int)
        self.start_time = time.time()

    async def startup(self):
        """
        Async startup handler for initializing resources.

        This method:
        - Initializes Redis connection for distributed queue
        - Sets up background task workers
        - Initializes cache managers
        - Prepares resource pools
        - Starts monitoring tasks
        """
        if self._initialized:
            self.logger.warning("ApiV2 already initialized, skipping startup")
            return

        self.logger.info("Starting SDNext v2 API services...")

        try:
            # Initialize Redis if enabled
            if config.redis_enabled:
                self.logger.info("Connecting to Redis...")
                self.redis_client = await get_redis_client()
                if self.redis_client:
                    self.logger.info("Redis connection established")
                else:
                    self.logger.warning("Redis connection failed, using in-memory fallback")

            # Initialize services
            await self._initialize_services()

            # Start background tasks
            await self._start_background_tasks()

            # Warm up caches
            await self._warm_caches()

            self._initialized = True
            self._startup_complete = True
            self.logger.info("SDNext v2 API started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start ApiV2: {e}")
            self.logger.error(traceback.format_exc())
            raise

    async def shutdown(self):
        """
        Async shutdown handler for cleaning up resources.
        """
        self.logger.info("Shutting down SDNext v2 API...")

        try:
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            # Close WebSocket connections
            await self.connection_manager.disconnect_all()

            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()

            # Cleanup services
            await self._cleanup_services()

            self._initialized = False
            self._startup_complete = False
            self.logger.info("SDNext v2 API shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())

    def register(self):
        """
        Register all v2 API routes and middleware.

        This method:
        - Registers all domain routers
        - Sets up middleware stack
        - Configures CORS settings
        - Adds WebSocket endpoints
        - Sets up error handlers
        """
        self.logger.info("Registering SDNext v2 API routes and middleware...")

        # Setup lifespan context manager
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()

        # Set lifespan if we own the app
        if self.owns_app:
            self.app.router.lifespan_context = lifespan

        # Add middleware (order matters - applied in reverse)
        self._setup_middleware()

        # Register domain routers
        self._register_routers()

        # Register WebSocket endpoints
        self._register_websockets()

        # Register error handlers
        self._register_error_handlers()

        # Register the main v2 router with the app
        self.app.include_router(self.router)

        # Register health check at root v2 path
        @self.router.get("/", tags=["System"])
        async def api_root():
            """Root endpoint for v2 API."""
            return {
                "name": "SDNext v2 API",
                "version": "2.0.0",
                "status": "operational" if self._startup_complete else "initializing",
                "uptime": time.time() - self.start_time,
                "endpoints": {"docs": "/sdapi/v2/docs", "redoc": "/sdapi/v2/redoc", "openapi": "/sdapi/v2/openapi.json", "websocket": "/sdapi/v2/ws", "events": "/sdapi/v2/events"},
            }

        self.logger.info("SDNext v2 API routes and middleware registered")

    def _setup_middleware(self):
        """Setup middleware stack."""

        # CORS middleware (most permissive for development)
        if not hasattr(self.app.state, "_cors_configured"):
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure based on cmd_opts if available
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self.app.state._cors_configured = True

        # GZip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Trusted host middleware (if configured)
        if hasattr(cmd_opts, "server_name") and cmd_opts.server_name:
            self.app.add_middleware(TrustedHostMiddleware, allowed_hosts=[cmd_opts.server_name, "localhost", "127.0.0.1"])

        # Custom middleware
        self.app.add_middleware(RequestLoggingMiddleware, logger=self.logger)
        self.app.add_middleware(MonitoringMiddleware, metrics=self.metrics)

        # Rate limiting (if enabled)
        if config.rate_limit_enabled:
            self.app.add_middleware(RateLimitMiddleware, requests=config.rate_limit_requests, period=config.rate_limit_period)

        # Authentication (if configured)
        if config.jwt_secret:
            self.app.add_middleware(JWTAuthMiddleware, secret=config.jwt_secret, algorithm=config.jwt_algorithm)

    def _register_routers(self):
        """Register all domain routers."""

        # System endpoints
        self.router.include_router(system_router, prefix="/system", tags=["System"])

        # Generation endpoints
        self.router.include_router(generation_router, prefix="/generation", tags=["Generation"])

        # Canvas endpoints
        self.router.include_router(canvas_router, prefix="/canvas", tags=["Canvas"])

        # Model management
        self.router.include_router(models_router, prefix="/models", tags=["Models"])

        # Queue management
        self.router.include_router(queue_router, prefix="/queue", tags=["Queue"])

        # Server-sent events
        self.router.include_router(events_router, prefix="/events", tags=["Events"])

        # Asset management
        self.router.include_router(assets_router, prefix="/assets", tags=["Assets"])

    def _register_websockets(self):
        """Register WebSocket endpoints."""

        @self.router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint for real-time updates."""
            client_id = await self.connection_manager.connect(websocket)
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()

                    # Process message (implement message handling)
                    response = await self._process_websocket_message(client_id, data)

                    # Send response
                    if response:
                        await self.connection_manager.send_personal_message(response, client_id)
            except Exception as e:
                self.logger.error(f"WebSocket error for client {client_id}: {e}")
            finally:
                await self.connection_manager.disconnect(client_id)

    def _register_error_handlers(self):
        """Register global error handlers."""

        @self.app.exception_handler(404)
        async def not_found_handler(request: Request, exc):
            return JSONResponse(status_code=404, content={"error": "Not Found", "message": f"The requested endpoint {request.url.path} was not found", "path": request.url.path})

        @self.app.exception_handler(500)
        async def internal_error_handler(request: Request, exc):
            self.logger.error(f"Internal error: {exc}")
            self.logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": "Internal Server Error", "message": "An unexpected error occurred", "request_id": request.state.request_id if hasattr(request.state, "request_id") else None})

    async def _initialize_services(self):
        """Initialize service instances."""

        # Import services dynamically to avoid circular imports
        try:
            from .services.cache_service import CacheService
            from .services.queue_service import QueueService
            from .services.storage_service import StorageService

            # Initialize cache service
            self.cache_service = CacheService(redis_client=self.redis_client)
            await self.cache_service.initialize()

            # Initialize queue service
            self.queue_service = QueueService(redis_client=self.redis_client, queue_lock=self.queue_lock)
            await self.queue_service.initialize()

            # Initialize storage service
            self.storage_service = StorageService()
            await self.storage_service.initialize()

        except ImportError as e:
            self.logger.warning(f"Some services not available: {e}")

    async def _cleanup_services(self):
        """Cleanup service instances."""

        if self.cache_service:
            await self.cache_service.cleanup()

        if self.queue_service:
            await self.queue_service.cleanup()

        if self.storage_service:
            await self.storage_service.cleanup()

    async def _start_background_tasks(self):
        """Start background tasks."""

        # Metrics collection task
        async def collect_metrics():
            while self._initialized:
                # Collect system metrics
                self.metrics["requests_total"] = self.metrics.get("requests_total", 0)
                self.metrics["active_connections"] = len(self.connection_manager.active_connections)

                # Sleep for collection interval
                await asyncio.sleep(30)

        # WebSocket heartbeat task
        async def websocket_heartbeat():
            while self._initialized:
                await self.connection_manager.broadcast({"type": "heartbeat", "timestamp": time.time()})
                await asyncio.sleep(config.ws_heartbeat_interval)

        # Queue processor task (if not using external workers)
        async def process_queue():
            while self._initialized:
                if self.queue_service:
                    await self.queue_service.process_next()
                await asyncio.sleep(0.1)  # Small delay to prevent spinning

        # Start tasks
        self._background_tasks = [asyncio.create_task(collect_metrics()), asyncio.create_task(websocket_heartbeat()), asyncio.create_task(process_queue())]

    async def _warm_caches(self):
        """Warm up caches with commonly used data."""

        if not self.cache_service:
            return

        try:
            # Cache model information
            from modules import shared

            if hasattr(shared, "sd_model") and shared.sd_model:
                await self.cache_service.set("current_model", {"name": getattr(shared.sd_model, "name", "unknown"), "hash": getattr(shared.sd_model, "hash", None)})

            # Cache system info
            import platform

            await self.cache_service.set("system_info", {"platform": platform.platform(), "python": platform.python_version(), "api_version": "2.0.0"})

        except Exception as e:
            self.logger.warning(f"Failed to warm caches: {e}")

    async def _process_websocket_message(self, client_id: str, message: str) -> Optional[Dict[str, Any]]:
        """
        Process incoming WebSocket message.

        Args:
            client_id: Unique client identifier
            message: Raw message string

        Returns:
            Response dictionary or None
        """
        try:
            import json

            data = json.loads(message)

            msg_type = data.get("type", "unknown")

            if msg_type == "ping":
                return {"type": "pong", "timestamp": time.time()}

            elif msg_type == "subscribe":
                # Subscribe to task updates
                task_id = data.get("task_id")
                if task_id:
                    await self.connection_manager.subscribe_to_task(client_id, task_id)
                    return {"type": "subscribed", "task_id": task_id}

            elif msg_type == "unsubscribe":
                # Unsubscribe from task updates
                task_id = data.get("task_id")
                if task_id:
                    await self.connection_manager.unsubscribe_from_task(client_id, task_id)
                    return {"type": "unsubscribed", "task_id": task_id}

            else:
                return {"type": "error", "message": f"Unknown message type: {msg_type}"}

        except json.JSONDecodeError:
            return {"type": "error", "message": "Invalid JSON"}
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
            return {"type": "error", "message": str(e)}

    # Public API methods for integration

    async def broadcast_update(self, update: Dict[str, Any]):
        """
        Broadcast an update to all connected clients.

        Args:
            update: Update dictionary to broadcast
        """
        await self.connection_manager.broadcast(update)

    async def send_task_update(self, task_id: str, update: Dict[str, Any]):
        """
        Send task update to subscribed clients.

        Args:
            task_id: Task identifier
            update: Update dictionary
        """
        await self.connection_manager.broadcast_task_update(task_id, update)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dictionary of metrics
        """
        return dict(self.metrics)
