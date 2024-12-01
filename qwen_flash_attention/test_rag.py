import asyncio
import httpx
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag():
    """Test RAG functionality through the API."""
    async with httpx.AsyncClient() as client:
        try:
            # Create a group
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
            
            logger.info("Creating group...")
            response = await client.post(
                "http://localhost:8001/v1/groups/create",
                json=group_config
            )
            response.raise_for_status()
            group_data = response.json()
            group_id = group_data["group_id"]
            logger.info(f"Created group: {group_id}")

            # Add test documents
            test_docs = [
                {
                    "text": "Python list comprehension is a concise way to create lists. "
                           "Example: [x for x in range(10) if x % 2 == 0] creates a list of even numbers.",
                    "metadata": {"type": "python", "topic": "list_comprehension"}
                },
                {
                    "text": "The with statement in Python is used for context management. "
                           "It automatically handles cleanup of resources like file handles.",
                    "metadata": {"type": "python", "topic": "context_managers"}
                }
            ]

            # Test chat completion
            chat_request = {
                "messages": [
                    {
                        "role": "user",
                        "content": "How do I use list comprehension in Python?"
                    }
                ],
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "temperature": 0.7,
                "max_tokens": 1024
            }

            logger.info("Testing chat completion...")
            response = await client.post(
                f"http://localhost:8001/v1/groups/{group_id}/chat",
                json=chat_request
            )
            response.raise_for_status()
            chat_result = response.json()
            
            logger.info("Chat Response:")
            logger.info(json.dumps(chat_result, indent=2))

        except Exception as e:
            logger.error(f"Error in test: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(test_rag())
