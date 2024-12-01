import pytest
import asyncio
import httpx
from typing import Dict, Any, AsyncGenerator
from fastapi.testclient import TestClient
from ..context_server import app as context_app
from ..model_server import app as model_app
from ..tools import default_registry

@pytest.fixture
async def context_client():
    async with httpx.AsyncClient(app=context_app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def model_client():
    async with httpx.AsyncClient(app=model_app, base_url="http://test") as client:
        yield client

class TestVSCodeIntegration:
    """Test MCP integration with VSCode extension."""

    async def test_code_completion_flow(self, context_client: httpx.AsyncClient):
        """Test complete code completion flow."""
        # Test model listing
        response = await context_client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()["data"]
        assert len(models) > 0
        assert models[0]["id"] == "Qwen2.5-Coder-1.5B-Instruct"

        # Test tool listing
        response = await context_client.get("/v1/tools")
        assert response.status_code == 200
        tools = response.json()["data"]
        assert len(tools) > 0
        assert any(tool["name"] == "readFile" for tool in tools)

        # Test code completion with file reading
        completion_request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Show me the contents of README.md"
                }
            ],
            "tools": ["readFile"],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        response = await context_client.post("/v1/chat/completions", json=completion_request)
        assert response.status_code == 200
        result = response.json()
        assert result["choices"][0]["message"]["tool_calls"] is not None

    async def test_code_editing_flow(self, context_client: httpx.AsyncClient):
        """Test code editing with file operations."""
        # Create a test file
        write_request = {
            "tool": "writeFile",
            "parameters": {
                "path": "test_file.py",
                "content": "def test_function():\n    pass"
            }
        }
        response = await context_client.post("/v1/tools/execute", json=write_request)
        assert response.status_code == 200
        assert response.json()["result"] is True

        # Request code improvement
        completion_request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Improve the function in test_file.py to add two numbers"
                }
            ],
            "tools": ["readFile", "writeFile"],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        response = await context_client.post("/v1/chat/completions", json=completion_request)
        assert response.status_code == 200
        result = response.json()
        assert result["choices"][0]["message"]["tool_calls"] is not None

        # Verify file was updated
        read_request = {
            "tool": "readFile",
            "parameters": {
                "path": "test_file.py"
            }
        }
        response = await context_client.post("/v1/tools/execute", json=read_request)
        assert response.status_code == 200
        content = response.json()["result"]
        assert "def add" in content

    async def test_knowledge_integration(self, context_client: httpx.AsyncClient):
        """Test knowledge base integration."""
        # Add knowledge
        add_request = {
            "tool": "addKnowledge",
            "parameters": {
                "text": "Python's sum() function can add numbers in a list",
                "metadata": {
                    "source": "documentation",
                    "type": "function",
                    "language": "python"
                }
            }
        }
        response = await context_client.post("/v1/tools/execute", json=add_request)
        assert response.status_code == 200

        # Query with knowledge enhancement
        completion_request = {
            "messages": [
                {
                    "role": "user",
                    "content": "How can I add numbers in a list in Python?"
                }
            ],
            "tools": ["searchKnowledge"],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        response = await context_client.post("/v1/chat/completions", json=completion_request)
        assert response.status_code == 200
        result = response.json()
        assert "sum()" in result["choices"][0]["message"]["content"]

    async def test_command_execution(self, context_client: httpx.AsyncClient):
        """Test command execution integration."""
        # List directory
        command_request = {
            "tool": "executeCommand",
            "parameters": {
                "command": "ls"
            }
        }
        response = await context_client.post("/v1/tools/execute", json=command_request)
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["returncode"] == 0
        assert len(result["stdout"]) > 0

    async def test_error_handling(self, context_client: httpx.AsyncClient):
        """Test error handling in VSCode context."""
        # Test invalid model
        completion_request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            "model": "invalid-model"
        }
        response = await context_client.post("/v1/chat/completions", json=completion_request)
        assert response.status_code == 400

        # Test invalid tool
        tool_request = {
            "tool": "invalidTool",
            "parameters": {}
        }
        response = await context_client.post("/v1/tools/execute", json=tool_request)
        assert response.status_code == 200
        assert response.json()["error"] is not None

    async def test_concurrent_operations(self, context_client: httpx.AsyncClient):
        """Test concurrent operations in VSCode context."""
        # Create multiple completion requests
        requests = []
        for i in range(5):
            requests.append(
                context_client.post("/v1/chat/completions", json={
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Count to {i}"
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048
                })
            )

        # Execute concurrently
        responses = await asyncio.gather(*requests)
        assert all(r.status_code == 200 for r in responses)
        assert len(set(r.json()["id"] for r in responses)) == 5  # All responses should be unique
