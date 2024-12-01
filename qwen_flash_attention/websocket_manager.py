import logging
import time
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.agent_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.add(websocket)
        if agent_id:
            self.agent_connections[agent_id] = websocket
            await self.broadcast_json({
                "type": "com.microsoft.autogen.message.state",
                "data": {
                    "event": "AGENT_REGISTER",
                    "agent_id": agent_id
                }
            })
        logger.info(f"WebSocket client connected{f' with agent_id {agent_id}' if agent_id else ''}")

    async def disconnect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        self.active_connections.remove(websocket)
        if agent_id and agent_id in self.agent_connections:
            del self.agent_connections[agent_id]
            await self.broadcast_json({
                "type": "com.microsoft.autogen.message.state",
                "data": {
                    "event": "AGENT_UNREGISTER",
                    "agent_id": agent_id
                }
            })
        logger.info(f"WebSocket client disconnected{f' with agent_id {agent_id}' if agent_id else ''}")

    async def broadcast_agent_message(self, agent_name: str, role: str, content: str):
        """Broadcast agent messages to all connected clients."""
        message = {
            "type": "agent_message",
            "agent": {
                "name": agent_name,
                "role": role
            },
            "content": content,
            "timestamp": int(time.time() * 1000)
        }
        logger.info(f"Broadcasting message from {agent_name} ({role})")
        await self.broadcast_json(message)

    async def broadcast_json(self, data: dict):
        """Broadcast JSON message to all connected clients."""
        for ws in self.active_connections.copy():
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.error(f"Error broadcasting JSON: {e}")
                await self.disconnect(ws)

    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send message to specific agent."""
        if agent_id in self.agent_connections:
            try:
                await self.agent_connections[agent_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to agent {agent_id}: {e}")
                ws = self.agent_connections[agent_id]
                await self.disconnect(ws, agent_id)

    async def handle_browser_event(self, event: Dict[str, Any]):
        """Handle browser-related events."""
        event_type = event.get("type")
        source = event.get("source")
        data = event.get("data", {})

        if event_type == "com.microsoft.autogen.message.state":
            if data.get("event") == "AGENT_REGISTER":
                agent_id = data.get("agent_id")
                if agent_id and source:
                    # Agent registration is handled in connect method
                    pass
            elif data.get("event") == "AGENT_UNREGISTER":
                agent_id = data.get("agent_id")
                if agent_id and source:
                    # Agent unregistration is handled in disconnect method
                    pass

        elif event_type == "com.microsoft.autogen.message.tool":
            if data.get("tool") == "browser_action":
                # Broadcast browser action result to all clients
                await self.broadcast_json({
                    "type": "com.microsoft.autogen.message.tool.browser",
                    "source": source,
                    "data": {
                        "action": data.get("action"),
                        "result": {
                            "screenshot": data.get("screenshot"),
                            "logs": data.get("logs", []),
                            "currentUrl": data.get("currentUrl"),
                            "currentMousePosition": data.get("currentMousePosition")
                        }
                    }
                })
