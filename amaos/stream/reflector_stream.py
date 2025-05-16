"""
ReflectorStream - WebSocket interface for AMAOS ReflectorNode events.

This module enables real-time streaming of ReflectorNode log entries
via FastAPI WebSocket endpoints.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from amaos.nodes.reflector_node import ReflectorNode, ReflectorLogEntry
from amaos.utils.context_logger import ContextAwareLogger


class StreamConfig(BaseModel):
    """Configuration for ReflectorStream."""
    
    host: str = "127.0.0.1"
    port: int = 8000
    log_buffer_size: int = 1000
    enable_history: bool = True
    cors_origins: List[str] = ["*"]
    log_level: str = "info"


class ReflectorStream:
    """WebSocket stream interface for ReflectorNode events.
    
    This class provides a FastAPI application that streams ReflectorNode
    log entries to WebSocket clients in real-time.
    """
    
    def __init__(self, config: Optional[StreamConfig] = None,
                 reflector_node: Optional[ReflectorNode] = None):
        """Initialize ReflectorStream.
        
        Args:
            config: Configuration for the stream
            reflector_node: ReflectorNode to stream from
        """
        self.config = config or StreamConfig()
        self.reflector_node = reflector_node
        self.logger = ContextAwareLogger("reflector_stream")
        
        # Initialize FastAPI app
        self.app = FastAPI(title="AMAOS ReflectorStream",
                          description="WebSocket interface for ReflectorNode events")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize WebSocket connection manager
        self.connection_manager = ConnectionManager()
        
        # Set up log buffer
        self.log_buffer: List[Dict[str, Any]] = []
        
        # Set up routes
        self._setup_routes()
        
        # Server instance - initialize as a property that will be set later
        self.server: Optional[uvicorn.Server] = None
        
    def _setup_routes(self) -> None:
        """Set up FastAPI routes and WebSocket endpoints."""
        
        @self.app.get("/")
        async def get_info() -> Dict[str, Any]:
            """Get information about the ReflectorStream."""
            if not self.reflector_node:
                return {"status": "no_reflector", "connected_clients": len(self.connection_manager.active_connections)}
                
            return {
                "status": "active",
                "connected_clients": len(self.connection_manager.active_connections),
                "buffer_size": len(self.log_buffer),
                "reflector_node": self.reflector_node.id,
                "log_count": self.reflector_node.metrics.total
            }
            
        @self.app.get("/logs")
        async def get_logs(
            limit: int = Query(20, ge=1, le=100),
            trace_id: Optional[str] = None,
            node_type: Optional[str] = None
        ) -> Dict[str, Any]:
            """Get recent logs with optional filtering."""
            if not self.reflector_node:
                raise HTTPException(status_code=503, detail="ReflectorNode not available")
                
            logs = self.reflector_node.get_logs(limit)
            
            # Apply filters if specified
            if trace_id or node_type:
                filtered_logs = []
                for log in logs:
                    # Check if log entry has metadata field
                    metadata = log.get("result", {}).get("metadata", {})
                    
                    # Apply trace_id filter
                    if trace_id and metadata.get("trace_id") != trace_id:
                        continue
                        
                    # Apply node_type filter
                    if node_type and log.get("node_type") != node_type:
                        continue
                        
                    filtered_logs.append(log)
                logs = filtered_logs
                
            return {
                "logs": logs,
                "count": len(logs),
                "filters": {
                    "trace_id": trace_id,
                    "node_type": node_type
                }
            }
            
        @self.app.get("/stats")
        async def get_stats() -> Dict[str, Any]:
            """Get ReflectorNode statistics."""
            if not self.reflector_node:
                raise HTTPException(status_code=503, detail="ReflectorNode not available")
                
            return self.reflector_node.get_stats()
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket,
                                    trace_id: Optional[str] = None,
                                    node_type: Optional[str] = None) -> None:
            """WebSocket endpoint for streaming log entries."""
            await self.connection_manager.connect(websocket)
            
            # Set up filters
            filters = {}
            if trace_id:
                filters["trace_id"] = trace_id
            if node_type:
                filters["node_type"] = node_type
                
            try:
                # Send recent logs if history is enabled
                if self.config.enable_history:
                    # Slice buffer from the end to get most recent logs
                    recent_logs = self.log_buffer[-50:]
                    
                    # Apply filters if necessary
                    if filters:
                        filtered_logs = []
                        for log in recent_logs:
                            if self._matches_filters(log, filters):
                                filtered_logs.append(log)
                        recent_logs = filtered_logs
                    
                    # Send history as a batch
                    if recent_logs:
                        await websocket.send_json({
                            "type": "history",
                            "logs": recent_logs,
                            "count": len(recent_logs)
                        })
                
                # Keep connection open and handle incoming messages
                while True:
                    # Wait for messages (allows disconnection)
                    data = await websocket.receive_text()
                    
                    # Handle client messages (e.g., filter updates)
                    try:
                        message = json.loads(data)
                        
                        if message.get("type") == "set_filters":
                            new_filters = message.get("filters", {})
                            filters = new_filters
                            
                            # Send confirmation
                            await websocket.send_json({
                                "type": "filters_updated",
                                "filters": filters
                            })
                    except Exception as e:
                        self.logger.error(f"Error processing client message: {e}")
                        
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
                
    def _matches_filters(self, log: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a log entry matches the specified filters.
        
        Args:
            log: Log entry to check
            filters: Filters to apply
            
        Returns:
            True if log matches all filters, False otherwise
        """
        for key, value in filters.items():
            # Special handling for trace_id which is nested in metadata
            if key == "trace_id":
                metadata = log.get("result", {}).get("metadata", {})
                if metadata.get("trace_id") != value:
                    return False
            elif log.get(key) != value:
                return False
                
        return True
        
    async def broadcast_log(self, log_entry: Union[ReflectorLogEntry, Dict[str, Any]]) -> None:
        """Broadcast a log entry to all connected WebSocket clients.
        
        Args:
            log_entry: Log entry to broadcast
        """
        # Convert to dict if necessary
        if isinstance(log_entry, ReflectorLogEntry):
            log_dict = log_entry.model_dump()
        else:
            log_dict = log_entry
            
        # Add to buffer, maintaining buffer size
        self.log_buffer.append(log_dict)
        if len(self.log_buffer) > self.config.log_buffer_size:
            self.log_buffer.pop(0)
            
        # Broadcast to all clients
        await self.connection_manager.broadcast({
            "type": "log",
            "timestamp": datetime.utcnow().isoformat(),
            "data": log_dict
        })
        
    async def start(self) -> None:
        """Start the ReflectorStream server."""
        if self.reflector_node:
            # Register handler for new log entries
            self.reflector_node.on_log_entry = self.broadcast_log
            self.logger.info(f"Registered log handler with ReflectorNode {self.reflector_node.id}")
            
        # Start server in a separate task
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level
        )
        
        # Create and initialize the server
        server = uvicorn.Server(config)
        self.server = server
        
        self.logger.info(f"Starting ReflectorStream on {self.config.host}:{self.config.port}")
        await self.server.serve()
        
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance.
        
        Returns:
            FastAPI application instance
        """
        return self.app


class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket) -> None:
        """Connect a WebSocket client.
        
        Args:
            websocket: WebSocket to connect
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client.
        
        Args:
            websocket: WebSocket to disconnect
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def broadcast(self, message: Any) -> None:
        """Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
                
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
