from fastapi import Request
import logging
import time
import json
from typing import Callable, Optional
from fastapi.responses import JSONResponse, Response
import traceback

# Configure logging with custom formatter
class CustomFormatter(logging.Formatter):
    grey = "\033[0;37m"
    reset = "\033[0m"
    format_str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    def format(self, record):
        formatter = logging.Formatter(self.format_str)
        record.msg = f"{self.grey}[HTTP] {record.msg}{self.reset}"
        return formatter.format(record)

# Configure middleware logger
logger = logging.getLogger("middleware")
logger.setLevel(logging.DEBUG)

# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

def format_log_message(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    body: Optional[str] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None
) -> str:
    """Format log message in a consistent way."""
    message_parts = [
        f"[{request_id}]",
        f"method={method}",
        f"path={path}",
        f"status={status_code}"
    ]
    
    if duration_ms is not None:
        message_parts.append(f"duration={duration_ms:.2f}ms")
    
    if error:
        message_parts.append(f"error={error}")
    
    message = " ".join(message_parts)
    
    if body:
        try:
            # Try to parse and format JSON
            json_body = json.loads(body) if isinstance(body, str) else body
            message += f"\nBody:\n{json.dumps(json_body, indent=2)}"
        except:
            # If not JSON, use raw body
            message += f"\nBody:\n{body}"
    
    return message

async def log_request_middleware(request: Request, call_next: Callable) -> Response:
    """Log request and response details."""
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}-{request.client.host}"
    
    # Log request
    try:
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            raw_body = await request.body()
            if raw_body:
                try:
                    body = json.loads(raw_body)
                except:
                    body = raw_body.decode()
            
            # Reset request body for downstream handlers
            async def receive():
                return {"type": "http.request", "body": raw_body}
            request._receive = receive
        
        logger.debug(format_log_message(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=0,  # Status not known yet
            body=body
        ))
    except Exception as e:
        logger.error(f"Error logging request: {str(e)}")
    
    # Process request
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        try:
            response_body = None
            if isinstance(response, JSONResponse):
                response_body = json.loads(response.body.decode())
            else:
                try:
                    response_body = response.body.decode()
                    try:
                        # Try to parse and format as JSON
                        response_body = json.loads(response_body)
                    except:
                        pass  # Keep as plain text
                except:
                    response_body = None
            
            logger.debug(format_log_message(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                body=response_body,
                duration_ms=duration_ms
            ))
        except Exception as e:
            logger.error(f"Error logging response: {str(e)}")
        
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        logger.error(format_log_message(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=500,
            error=error_msg,
            duration_ms=duration_ms
        ))
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        error_response = {
            "detail": error_msg,
            "traceback": traceback.format_exc()
        }
        return JSONResponse(
            status_code=500,
            content=error_response
        )
