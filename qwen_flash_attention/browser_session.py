import asyncio
import json
import logging
from typing import Dict, Any, Optional
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)

class BrowserSession:
    def __init__(self, agent_id: str, ws_url: str = "ws://localhost:8000/ws"):
        self.agent_id = agent_id
        self.ws_url = ws_url
        self.ws: Optional[WebSocketClientProtocol] = None
        self.browser_state: Dict[str, Any] = {
            "currentUrl": None,
            "currentMousePosition": None,
            "screenshot": None,
            "logs": []
        }

    async def connect(self):
        """Connect to WebSocket server and register agent."""
        if not self.ws:
            try:
                self.ws = await websockets.connect(self.ws_url)
                # Register agent
                await self.ws.send(json.dumps({
                    "type": "com.microsoft.autogen.message.state",
                    "source": self.agent_id,
                    "data": {
                        "event": "AGENT_REGISTER",
                        "agent_id": self.agent_id
                    }
                }))
                logger.info(f"Browser session connected for agent {self.agent_id}")
            except Exception as e:
                logger.error(f"Failed to connect browser session: {e}")
                raise

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.ws:
            try:
                # Unregister agent
                await self.ws.send(json.dumps({
                    "type": "com.microsoft.autogen.message.state",
                    "source": self.agent_id,
                    "data": {
                        "event": "AGENT_UNREGISTER",
                        "agent_id": self.agent_id
                    }
                }))
                await self.ws.close()
                self.ws = None
                logger.info(f"Browser session disconnected for agent {self.agent_id}")
            except Exception as e:
                logger.error(f"Error disconnecting browser session: {e}")
                raise

    async def send_browser_event(self, action: str, **kwargs) -> Dict[str, Any]:
        """Send browser action event through WebSocket."""
        if not self.ws:
            await self.connect()

        try:
            event = {
                "type": "com.microsoft.autogen.message.tool",
                "source": self.agent_id,
                "data": {
                    "tool": "browser_action",
                    "action": action,
                    **kwargs
                }
            }
            await self.ws.send(json.dumps(event))

            # Wait for response
            while True:
                response = await self.ws.recv()
                event_data = json.loads(response)
                
                # Only process browser tool responses for this agent
                if (event_data["type"] == "com.microsoft.autogen.message.tool.browser" and 
                    event_data["source"] == self.agent_id):
                    self.browser_state.update(event_data["data"]["result"])
                    return event_data["data"]["result"]

        except Exception as e:
            logger.error(f"Error sending browser event: {e}")
            raise

    async def launch(self, url: str) -> Dict[str, Any]:
        """Launch browser with URL."""
        logger.info(f"Launching browser with URL: {url}")
        return await self.send_browser_event("launch", url=url)

    async def click(self, coordinate: str) -> Dict[str, Any]:
        """Click at coordinates."""
        logger.info(f"Clicking at coordinates: {coordinate}")
        return await self.send_browser_event("click", coordinate=coordinate)

    async def type(self, text: str) -> Dict[str, Any]:
        """Type text."""
        logger.info(f"Typing text: {text}")
        return await self.send_browser_event("type", text=text)

    async def scroll_down(self) -> Dict[str, Any]:
        """Scroll down."""
        logger.info("Scrolling down")
        return await self.send_browser_event("scroll_down")

    async def scroll_up(self) -> Dict[str, Any]:
        """Scroll up."""
        logger.info("Scrolling up")
        return await self.send_browser_event("scroll_up")

    async def close(self) -> Dict[str, Any]:
        """Close browser."""
        logger.info("Closing browser")
        try:
            result = await self.send_browser_event("close")
            await self.disconnect()
            return result
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
            raise

    def get_state(self) -> Dict[str, Any]:
        """Get current browser state."""
        return self.browser_state.copy()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
