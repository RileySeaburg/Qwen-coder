import pytest
import requests
import json
import logging
import time
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8000"

def wait_for_server(url: str = SERVER_URL, max_retries: int = 10, delay: float = 2.0) -> bool:
    """Wait for server to be ready."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/v1/models")
            if response.status_code == 200:
                return True
        except:
            pass
        logger.info(f"Waiting for server... ({i+1}/{max_retries})")
        time.sleep(delay)
    return False

def log_response(response: requests.Response) -> None:
    """Log response details."""
    try:
        data = response.json()
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Response: {json.dumps(data, indent=2)}")
    except:
        logger.info(f"Raw response: {response.text}")

@pytest.mark.asyncio
async def test_models_endpoint():
    """Test /v1/models endpoint."""
    response = requests.get(f"{SERVER_URL}/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    log_response(response)

@pytest.mark.asyncio
async def test_tools_endpoint():
    """Test /v1/tools endpoint."""
    response = requests.get(f"{SERVER_URL}/v1/tools")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    log_response(response)

@pytest.mark.asyncio
async def test_chat_completion():
    """Test /v1/chat/completions endpoint."""
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function to calculate factorial."
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    log_response(response)

@pytest.mark.asyncio
async def test_tool_execution():
    """Test /v1/tools/execute endpoint."""
    # First create a test file
    write_response = requests.post(
        f"{SERVER_URL}/v1/tools/execute",
        json={
            "tool": "writeFile",
            "parameters": {
                "path": "test.py",
                "content": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
            }
        }
    )
    assert write_response.status_code == 200
    log_response(write_response)

    # Then read it back
    read_response = requests.post(
        f"{SERVER_URL}/v1/tools/execute",
        json={
            "tool": "readFile",
            "parameters": {
                "path": "test.py"
            }
        }
    )
    assert read_response.status_code == 200
    data = read_response.json()
    assert "result" in data
    assert "factorial" in data["result"]
    log_response(read_response)

@pytest.mark.asyncio
async def test_chat_with_tools():
    """Test chat completion with tool usage."""
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Read the contents of test.py and improve the error handling."
                }
            ],
            "tools": ["readFile", "writeFile"],
            "temperature": 0.7,
            "max_tokens": 2048
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    message = data["choices"][0]["message"]
    assert "tool_calls" in message
    log_response(response)

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling."""
    # Test invalid tool
    response = requests.post(
        f"{SERVER_URL}/v1/tools/execute",
        json={
            "tool": "invalidTool",
            "parameters": {}
        }
    )
    data = response.json()
    assert "error" in data
    log_response(response)

    # Test missing parameters
    response = requests.post(
        f"{SERVER_URL}/v1/tools/execute",
        json={
            "tool": "readFile",
            "parameters": {}
        }
    )
    data = response.json()
    assert "error" in data
    log_response(response)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
