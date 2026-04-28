"""Cloud WebSocket channel.

Single endpoint `/sdapi/v1/ws`. Cloud job runner threads publish progress
events via `publish(event)`; the call schedules `manager.broadcast_async(...)`
on the FastAPI server's event loop using `asyncio.run_coroutine_threadsafe`.

The loop is captured at FastAPI's `startup` event. Calls before that (very
early CLI requests, framework probe paths) are no-ops — REST polling at
`GET /sdapi/v1/cloud/jobs/{id}` is the source of truth, the WebSocket is
best-effort.

Server backend assumption: production runs on `UvicornServer`
(asyncio/uvloop). The `HypercornServer` Trio path at `modules/api/api.py:201`
is commented out; if someone uncomments it later, `asyncio.get_running_loop`
will raise at startup, which is the correct loud failure.
"""
from __future__ import annotations
import asyncio
from typing import Optional, TYPE_CHECKING
from starlette.websockets import WebSocket, WebSocketState
from modules.logger import log


if TYPE_CHECKING:
    from fastapi import FastAPI


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        try:
            self.active.remove(ws)
        except ValueError:
            pass

    async def broadcast_async(self, event: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.active):
            if ws.client_state != WebSocketState.CONNECTED:
                dead.append(ws)
                continue
            try:
                await ws.send_json(event)
            except Exception as e:
                log.debug(f'Cloud WS: broadcast failed: {e}')
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


def publish(event: dict) -> None:
    """Sync-safe broadcaster, callable from any thread.

    No-ops if the FastAPI loop hasn't been captured yet or has been closed.
    """
    loop = manager.loop
    if loop is None or loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(manager.broadcast_async(event), loop)
    except RuntimeError as e:
        log.debug(f'Cloud WS: publish failed: {e}')


def register_api(app: FastAPI) -> None:
    @app.websocket('/sdapi/v1/ws')
    async def ws_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except Exception:
            pass
        finally:
            manager.disconnect(ws)

    @app.on_event('startup')
    async def capture_loop() -> None:
        manager.loop = asyncio.get_running_loop()
