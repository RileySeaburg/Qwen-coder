import logging
import gc
import torch
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .model import QwenModel
from .tool_model import ToolModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen API",
    description="API for Qwen language model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
tool_model = None
qwen_model = None

def clear_gpu_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

@app.on_event("startup")
async def startup():
    global tool_model, qwen_model
    
    try:
        # Clear GPU memory before starting
        clear_gpu_memory()
        
        # Initialize Qwen model
        logger.info("Initializing Qwen model...")
        try:
            qwen_model = QwenModel()
            logger.info("Qwen model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {str(e)}")
            qwen_model = None
            
    except Exception as e:
        logger.error(f"Error in startup: {str(e)}")
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if qwen_model is None:
                await websocket.send_text("Error: Model not initialized")
                continue
            try:
                response = qwen_model.generate(data)
                await websocket.send_text(response)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                await websocket.send_text(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Chat completion endpoint"""
    if qwen_model is None:
        return JSONResponse({
            "error": "Model not initialized"
        }, status_code=503)
    try:
        # Get request body as JSON
        body = await request.json()
        # Get the last message from the conversation
        last_message = body["messages"][-1]["content"]
        response = qwen_model.generate(last_message)
        return JSONResponse({
            "response": response
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse({
            "error": str(e)
        }, status_code=500)
