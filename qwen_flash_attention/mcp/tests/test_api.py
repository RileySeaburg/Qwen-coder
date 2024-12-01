import requests
import json
import logging
import time
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mcp_api():
    """Test MCP API endpoints."""
    try:
        # Test model listing
        logger.info("Testing model listing...")
        models_response = requests.get("http://localhost:8000/v1/models")
        models_response.raise_for_status()
        logger.info("Models available:")
        logger.info(json.dumps(models_response.json(), indent=2))

        # Test tool listing
        logger.info("\nTesting tool listing...")
        tools_response = requests.get("http://localhost:8000/v1/tools")
        tools_response.raise_for_status()
        logger.info("Tools available:")
        logger.info(json.dumps(tools_response.json(), indent=2))

        # Test chat completion without tools
        logger.info("\nTesting standard chat completion...")
        chat_response = requests.post(
            "http://localhost:8000/v1/chat/completions",
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
        chat_response.raise_for_status()
        logger.info("Chat response received:")
        logger.info(json.dumps(chat_response.json(), indent=2))

        # Test file operations with tools
        logger.info("\nTesting file operations...")
        file_chat_response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a file named test.py with a factorial function and proper error handling."
                    }
                ],
                "tools": ["readFile", "writeFile"],
                "temperature": 0.7,
                "max_tokens": 2048
            }
        )
        file_chat_response.raise_for_status()
        logger.info("File operation response received:")
        logger.info(json.dumps(file_chat_response.json(), indent=2))

        # Test knowledge base operations
        logger.info("\nTesting knowledge base operations...")
        knowledge_chat_response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Add information about Python's factorial function to the knowledge base."
                    }
                ],
                "tools": ["addKnowledge", "searchKnowledge"],
                "temperature": 0.7,
                "max_tokens": 2048
            }
        )
        knowledge_chat_response.raise_for_status()
        logger.info("Knowledge base operation response received:")
        logger.info(json.dumps(knowledge_chat_response.json(), indent=2))

        # Test command execution
        logger.info("\nTesting command execution...")
        command_chat_response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "List files in the current directory."
                    }
                ],
                "tools": ["executeCommand"],
                "temperature": 0.7,
                "max_tokens": 2048
            }
        )
        command_chat_response.raise_for_status()
        logger.info("Command execution response received:")
        logger.info(json.dumps(command_chat_response.json(), indent=2))

        # Test tool execution directly
        logger.info("\nTesting direct tool execution...")
        tool_response = requests.post(
            "http://localhost:8000/v1/tools/execute",
            json={
                "tool": "readFile",
                "parameters": {
                    "path": "test.py"
                }
            }
        )
        tool_response.raise_for_status()
        logger.info("Tool execution response received:")
        logger.info(json.dumps(tool_response.json(), indent=2))

        # Test error handling
        logger.info("\nTesting error handling...")
        error_response = requests.post(
            "http://localhost:8000/v1/tools/execute",
            json={
                "tool": "nonexistentTool",
                "parameters": {}
            }
        )
        logger.info("Error response received:")
        logger.info(json.dumps(error_response.json(), indent=2))

        logger.info("\nAll tests passed successfully!")

    except Exception as e:
        logger.error(f"Error testing API: {e}")
        raise

def verify_response(response: Dict[str, Any]) -> None:
    """Verify response structure and content."""
    if "error" in response:
        logger.warning(f"Response contains error: {response['error']}")
        return

    if "choices" in response:
        for choice in response["choices"]:
            if "message" in choice:
                message = choice["message"]
                if "tool_calls" in message:
                    logger.info("Tool calls found:")
                    for tool_call in message["tool_calls"]:
                        logger.info(f"Tool: {tool_call['name']}")
                        logger.info(f"Parameters: {tool_call['parameters']}")

    if "usage" in response:
        logger.info(f"Token usage: {response['usage']}")

if __name__ == "__main__":
    test_mcp_api()
