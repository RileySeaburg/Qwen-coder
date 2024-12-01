import logging
from fastapi import FastAPI, WebSocket
from .websocket_manager import WebSocketManager
from .model import QwenModel
from .tool_model import ToolModel
from shared_embeddings.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize WebSocket manager
ws_manager = WebSocketManager()
logger.info("WebSocket manager initialized")

# Initialize vector store
vector_store = VectorStore()

# Initialize models
tool_model = None
qwen_model = None

@app.on_event("startup")
async def startup():
    global tool_model, qwen_model
    
    try:
        # Initialize ToolACE model first (smaller model with quantization)
        logger.info("Initializing ToolACE model...")
        tool_model = ToolModel()
        
        # Then initialize Qwen model
        logger.info("Initializing Qwen model...")
        qwen_model = QwenModel()
        
    except Exception as e:
        logger.error(f"Error in startup: {str(e)}")
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process message and get response
            response = await process_message(data)
            # Send response back
            await websocket.send_text(response)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await ws_manager.disconnect(websocket)

async def process_message(message: str) -> str:
    # Add your message processing logic here
    return f"Received: {message}"
