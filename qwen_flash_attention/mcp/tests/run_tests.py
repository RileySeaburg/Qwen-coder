import pytest
import asyncio
import os
import sys
import logging
import subprocess
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_server(cmd: List[str], log_file: str) -> subprocess.Popen:
    """Start a server process."""
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        )
        return process

def stop_server(server: subprocess.Popen) -> None:
    """Stop a server process."""
    try:
        server.terminate()
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()

async def main():
    """Run all tests."""
    servers = []
    try:
        # Start model server
        logger.info("Starting model server...")
        model_server = start_server(
            ["python", "-m", "uvicorn", "qwen_flash_attention.mcp.model_server:app", "--host", "127.0.0.1", "--port", "8001", "--log-level", "error"],
            "logs/model_server.log"
        )
        servers.append(model_server)

        # Start context server
        logger.info("Starting context server...")
        context_server = start_server(
            ["python", "-m", "uvicorn", "qwen_flash_attention.mcp.context_server:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "error"],
            "logs/context_server.log"
        )
        servers.append(context_server)

        # Wait for servers to start
        await asyncio.sleep(5)
        
        # Run tests
        logger.info("Running tests...")
        test_args = [
            "-v",  # Verbose output
            "--asyncio-mode=auto",  # Handle async tests
            "qwen_flash_attention/mcp/tests/test_functional.py",
            "qwen_flash_attention/mcp/tests/test_integration.py",
            "qwen_flash_attention/mcp/tests/test_system.py",
            "qwen_flash_attention/mcp/tests/test_api.py"
        ]
        
        # Add coverage if requested
        if "--coverage" in sys.argv:
            test_args.extend([
                "--cov=qwen_flash_attention/mcp",
                "--cov-report=term-missing",
                "--cov-report=html"
            ])
        
        result = pytest.main(test_args)
        return result

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1

    finally:
        # Stop servers
        logger.info("Stopping servers...")
        for server in servers:
            stop_server(server)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Run tests
    result = asyncio.run(main())
    sys.exit(result)
