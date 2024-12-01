import asyncio
import json
import logging
from typing import Dict, Any, Optional, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

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
            await self.broadcast({
                "type": "com.microsoft.autogen.message.state",
                "data": {
                    "event": "AGENT_REGISTER",
                    "agent_id": agent_id
                }
            })

    async def disconnect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        self.active_connections.remove(websocket)
        if agent_id and agent_id in self.agent_connections:
            del self.agent_connections[agent_id]
            await self.broadcast({
                "type": "com.microsoft.autogen.message.state",
                "data": {
                    "event": "AGENT_UNREGISTER",
                    "agent_id": agent_id
                }
            })

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send message to specific agent."""
        if agent_id in self.agent_connections:
            try:
                await self.agent_connections[agent_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to agent {agent_id}: {e}")

class BrowserEvent(BaseModel):
    type: str
    source: str
    data: Dict[str, Any]

def create_websocket_app() -> FastAPI:
    app = FastAPI()
    manager = WebSocketManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        agent_id = None
        try:
            await manager.connect(websocket)
            
            while True:
                message = await websocket.receive_json()
                event = BrowserEvent(**message)

                if event.type == "com.microsoft.autogen.message.state":
                    if event.data.get("event") == "AGENT_REGISTER":
                        agent_id = event.data.get("agent_id")
                        await manager.connect(websocket, agent_id)
                    elif event.data.get("event") == "AGENT_UNREGISTER":
                        await manager.disconnect(websocket, agent_id)

                elif event.type == "com.microsoft.autogen.message.chat":
                    # Broadcast chat messages to all clients
                    await manager.broadcast(message)

                elif event.type == "com.microsoft.autogen.message.tool":
                    # Handle tool-related events
                    if event.data.get("tool") == "browser_action":
                        # Browser tool events get special handling
                        await manager.broadcast({
                            "type": "com.microsoft.autogen.message.tool.browser",
                            "source": event.source,
                            "data": {
                                "action": event.data.get("action"),
                                "result": {
                                    "screenshot": event.data.get("screenshot"),
                                    "logs": event.data.get("logs", []),
                                    "currentUrl": event.data.get("currentUrl"),
                                    "currentMousePosition": event.data.get("currentMousePosition")
                                }
                            }
                        })
                    else:
                        # Other tool events are broadcast normally
                        await manager.broadcast(message)

        except WebSocketDisconnect:
            if agent_id:
                await manager.disconnect(websocket, agent_id)
            else:
                await manager.disconnect(websocket)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if agent_id:
                await manager.disconnect(websocket, agent_id)
            else:
                await manager.disconnect(websocket)

    return app

# Create the WebSocket app
websocket_app = create_websocket_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(websocket_app, host="0.0.0.0", port=8001)
