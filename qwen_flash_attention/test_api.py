import asyncio
import httpx
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Longer timeout for model operations
TIMEOUT = 300.0  # 5 minutes

async def test_chat_completion():
    """Test basic chat completion."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        try:
            request = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a Python function to calculate factorial."
                    }
                ],
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "temperature": 0.7
            }
            
            logger.info("Testing chat completion...")
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                json=request
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("Chat Response:")
            logger.info(json.dumps(result, indent=2))
            return result

        except Exception as e:
            logger.error(f"Error in chat completion test: {e}")
            raise

async def test_group_chat():
    """Test group chat functionality."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        try:
            # First register a group
            group_config = {
                "agents": [
                    {
                        "name": "coder",
                        "role": "assistant",
                        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        "systemPrompt": "You are a helpful coding assistant."
                    }
                ],
                "teamType": "round_robin"
            }
            
            logger.info("Registering group...")
            response = await client.post(
                "http://localhost:8000/v1/group/register",
                json=group_config
            )
            response.raise_for_status()
            group_data = response.json()
            group_id = group_data["group_id"]
            
            # Now test group chat
            request = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a Python class for a binary search tree."
                    }
                ],
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "temperature": 0.7
            }
            
            logger.info("Testing group chat...")
            response = await client.post(
                f"http://localhost:8000/v1/groups/{group_id}/chat",
                json=request
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("Group Chat Response:")
            logger.info(json.dumps(result, indent=2))
            return result

        except Exception as e:
            logger.error(f"Error in group chat test: {e}")
            raise

async def test_autogen_team():
    """Test autogen team functionality."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        try:
            request = {
                "agents": [
                    {
                        "name": "coder",
                        "role": "assistant",
                        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        "systemPrompt": "You are a helpful coding assistant."
                    }
                ],
                "task": "Write a Python function to implement quicksort."
            }
            
            logger.info("Testing autogen team...")
            response = await client.post(
                "http://localhost:8000/v1/autogen/team/solve",
                json=request
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("Autogen Team Response:")
            logger.info(json.dumps(result, indent=2))
            return result

        except Exception as e:
            logger.error(f"Error in autogen team test: {e}")
            raise

async def main():
    """Run all tests."""
    try:
        # Test basic chat completion
        await test_chat_completion()
        
        # Test group chat
        await test_group_chat()
        
        # Test autogen team
        await test_autogen_team()
        
    except Exception as e:
        logger.error(f"Error in tests: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
