# SDNext v2 API

A modern, async-first API implementation for SDNext that provides enhanced performance, real-time updates, and improved scalability.

## Features

- **Async-First Design**: Built with FastAPI and async/await throughout
- **Real-time Updates**: WebSocket and Server-Sent Events support
- **Enhanced Queue Management**: Redis-backed distributed queue with priority support
- **Multi-tier Caching**: In-memory, Redis, and semantic similarity caching
- **UI Agnostic**: No Gradio dependencies, works with any frontend
- **Backward Compatible**: Runs alongside v1 API with adapter support

## Architecture

```
modules/api/v2/
├── api.py           # Main ApiV2 class
├── config.py        # Configuration management
├── dependencies.py  # Dependency injection
├── routers/         # Domain-specific routers
│   ├── system.py    # System information endpoints
│   ├── generation.py # Image generation endpoints
│   ├── canvas.py    # Canvas operations
│   ├── models.py    # Model management
│   └── queue.py     # Queue management
├── models/          # Pydantic data models
│   ├── generation.py # Generation request/response models
│   ├── system.py    # System info models
│   └── common.py    # Shared models
├── middleware/      # Custom middleware
│   ├── auth.py      # Authentication
│   └── rate_limit.py # Rate limiting
├── services/        # Business logic services
│   ├── queue_service.py    # Queue management
│   ├── cache_service.py    # Caching logic
│   └── storage_service.py  # Asset storage
├── utils/           # Utility functions
│   └── file_utils.py # File operations
└── websockets/      # WebSocket handlers
    └── manager.py   # Connection management
```

## Installation

The v2 API is included with SDNext and requires no additional installation.
Required dependencies will be installed automatically.

## Configuration

Configuration can be set via environment variables or the `.env.v2` file:

```bash
# API Settings
SDNEXT_V2_API_HOST=0.0.0.0
SDNEXT_V2_API_PORT=8000

# Redis Settings (optional)
SDNEXT_V2_REDIS_ENABLED=true
SDNEXT_V2_REDIS_HOST=localhost
SDNEXT_V2_REDIS_PORT=6379

# Cache Settings
SDNEXT_V2_CACHE_ENABLED=true
SDNEXT_V2_CACHE_TTL=3600
```

## Usage

The v2 API will be available at `/sdapi/v2/` endpoints once enabled.

### Example Requests

```python
# Text to Image Generation
response = requests.post(
    "http://localhost:7860/sdapi/v2/generation/txt2img",
    json={
        "prompt": "a beautiful landscape",
        "width": 512,
        "height": 512,
        "steps": 20
    }
)

# WebSocket connection for real-time updates
ws = websocket.WebSocket()
ws.connect("ws://localhost:7860/sdapi/v2/ws")
```
