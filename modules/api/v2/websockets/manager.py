"""
WebSocket connection manager for API v2

This module handles WebSocket connections, broadcasting,
and task-specific subscriptions.
"""

import asyncio
import json
import uuid
from typing import Dict, Set, List, Any, Optional
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
import logging


class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting.
    
    Features:
    - Connection lifecycle management
    - Room-based broadcasting
    - Task-specific subscriptions
    - Automatic cleanup on disconnect
    """
    
    def __init__(self):
        """Initialize the connection manager."""
        # Active WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Task subscriptions (task_id -> set of client_ids)
        self.task_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Client to tasks mapping (client_id -> set of task_ids)
        self.client_tasks: Dict[str, Set[str]] = defaultdict(set)
        
        # Room subscriptions (room_name -> set of client_ids)
        self.room_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Client to rooms mapping (client_id -> set of room_names)
        self.client_rooms: Dict[str, Set[str]] = defaultdict(set)
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            
        Returns:
            Unique client ID for this connection
        """
        await websocket.accept()
        
        # Generate unique client ID
        client_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[client_id] = websocket
        
        # Send connection confirmation
        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id
            },
            client_id
        )
        
        self.logger.info(f"Client {client_id} connected")
        return client_id
    
    async def disconnect(self, client_id: str):
        """
        Handle WebSocket disconnection.
        
        Args:
            client_id: The client ID to disconnect
        """
        if client_id in self.active_connections:
            # Clean up task subscriptions
            for task_id in self.client_tasks[client_id]:
                self.task_subscriptions[task_id].discard(client_id)
            del self.client_tasks[client_id]
            
            # Clean up room subscriptions
            for room_name in self.client_rooms[client_id]:
                self.room_subscriptions[room_name].discard(client_id)
            del self.client_rooms[client_id]
            
            # Remove connection
            del self.active_connections[client_id]
            
            self.logger.info(f"Client {client_id} disconnected")
    
    async def disconnect_all(self):
        """Disconnect all active connections."""
        client_ids = list(self.active_connections.keys())
        for client_id in client_ids:
            try:
                await self.active_connections[client_id].close()
            except Exception as e:
                self.logger.error(f"Error closing connection {client_id}: {e}")
            await self.disconnect(client_id)
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """
        Send a message to a specific client.
        
        Args:
            message: Message dictionary to send
            client_id: Target client ID
        """
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                self.logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message dictionary to broadcast
        """
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def subscribe_to_task(self, client_id: str, task_id: str):
        """Subscribe a client to task updates."""
        self.task_subscriptions[task_id].add(client_id)
        self.client_tasks[client_id].add(task_id)
        self.logger.debug(f"Client {client_id} subscribed to task {task_id}")
    
    async def unsubscribe_from_task(self, client_id: str, task_id: str):
        """Unsubscribe a client from task updates."""
        self.task_subscriptions[task_id].discard(client_id)
        self.client_tasks[client_id].discard(task_id)
        self.logger.debug(f"Client {client_id} unsubscribed from task {task_id}")
    
    async def broadcast_task_update(self, task_id: str, update: Dict[str, Any]):
        """
        Broadcast an update to all clients subscribed to a task.
        
        Args:
