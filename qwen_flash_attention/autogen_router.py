import logging
import httpx
import os
from typing import Dict, Any, List
import re
from .schemas import ChatCompletionRequest, Agent, AgentTeamConfig
from shared_embeddings.vector_store import VectorStore, VectorDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QWEN_API_URL = "http://localhost:8000/v1/chat/completions"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

class AutoGenRouter:
    def __init__(self):
        self.vector_store = VectorStore(
            mongodb_url="mongodb://localhost:27017",
            database_name="autogen_vectors",
            collection_name="conversations"
        )

    async def initialize(self):
        """Initialize vector store."""
        await self.vector_store.initialize()

    def determine_task_type(self, message: str) -> str:
        """Determine the type of task from the message."""
        message = message.lower()
        
        patterns = {
            "code": r"(code|program|function|class|implement|debug|fix|refactor)",
            "architecture": r"(design|architect|structure|pattern|system)",
            "review": r"(review|analyze|assess|evaluate|check)",
            "test": r"(test|unit test|integration test|qa|quality)",
            "documentation": r"(document|comment|explain|describe)",
            "devops": r"(deploy|ci/cd|pipeline|docker|kubernetes)",
            "database": r"(database|sql|query|schema|model)",
            "security": r"(security|auth|encrypt|protect|vulnerability)"
        }
        
        matches = {task: bool(re.search(pattern, message)) 
                for task, pattern in patterns.items()}
        
        matched_tasks = [task for task, matched in matches.items() if matched]
        return matched_tasks[0] if matched_tasks else "general"

    async def store_conversation(self, messages: List[Dict[str, str]], agent: Agent):
        """Store conversation in vector store for future context."""
        try:
            # Combine messages into a single document
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            ])

            doc = VectorDocument(
                text=conversation_text,
                metadata={
                    "agent": agent.name,
                    "task_type": self.determine_task_type(messages[-1]["content"]),
                    "timestamp": messages[-1].get("timestamp", None)
                }
            )

            await self.vector_store.add_documents([doc], source="conversation_history")
            logger.info(f"Stored conversation for agent {agent.name}")

        except Exception as e:
            logger.error(f"Error storing conversation: {e}")

    async def get_relevant_context(self, message: str, agent: Agent) -> List[str]:
        """Retrieve relevant context from previous conversations."""
        try:
            task_type = self.determine_task_type(message)
            results = await self.vector_store.search_similar(
                query=message,
                num_results=3,
                metadata_filter={"task_type": task_type}
            )

            context = []
            for result in results:
                if result["score"] > 0.7:  # Only include highly relevant context
                    context.append(result["text"])

            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    async def route_to_qwen(self, request: ChatCompletionRequest, agent: Agent) -> Dict[str, Any]:
        """Route request to local Qwen API with vector store context."""
        try:
            logger.info(f"Routing to Qwen API for agent {agent.name}")

            # Get relevant context
            last_message = request.messages[-1].content
            context = await self.get_relevant_context(last_message, agent)

            # Extract system messages and context
            system_messages = []
            conversation_messages = []

            for msg in request.messages:
                if msg.role == "system":
                    system_messages.append(msg)
                else:
                    conversation_messages.append(msg)

            # Combine messages with context
            messages = []
            
            # Add system prompts first
            messages.extend([
                {
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name
                }
                for msg in system_messages
            ])

            # Add retrieved context
            if context:
                messages.append({
                    "role": "system",
                    "content": "Relevant information from knowledge base:\n" + "\n---\n".join(context),
                    "name": "context"
                })

            # Add conversation history
            messages.extend([
                {
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name
                }
                for msg in conversation_messages
            ])

            logger.info(f"Sending request to Qwen API with {len(messages)} messages")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    QWEN_API_URL,
                    json={
                        "messages": messages,
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "stream": request.stream
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Store conversation for future context
                await self.store_conversation(messages, agent)

                # Format response with agent name
                result["choices"][0]["message"]["content"] = f"[{agent.name}] {result['choices'][0]['message']['content']}"
                result["choices"][0]["message"]["name"] = agent.name

                return result

        except Exception as e:
            logger.error(f"Error routing to Qwen: {e}")
            raise

    async def route_to_claude(self, request: ChatCompletionRequest, agent: Agent) -> Dict[str, Any]:
        """Route request to Claude API with vector store context."""
        if not CLAUDE_API_KEY:
            raise ValueError("CLAUDE_API_KEY environment variable not set")

        try:
            logger.info(f"Routing to Claude API for agent {agent.name}")

            # Get relevant context
            last_message = request.messages[-1].content
            context = await self.get_relevant_context(last_message, agent)

            # Extract system messages
            system_messages = []
            conversation_messages = []

            for msg in request.messages:
                if msg.role == "system":
                    system_messages.append(msg)
                else:
                    conversation_messages.append(msg)

            # Format messages for Claude
            messages = []
            
            # Add system prompts first
            messages.extend([
                {
                    "role": "assistant",
                    "content": msg.content
                }
                for msg in system_messages
            ])

            # Add retrieved context
            if context:
                messages.append({
                    "role": "assistant",
                    "content": "Relevant information from knowledge base:\n" + "\n---\n".join(context)
                })
            
            # Add conversation history
            for msg in conversation_messages:
                messages.append({
                    "role": "user" if msg.role == "user" else "assistant",
                    "content": f"[{msg.name}]: {msg.content}" if msg.name else msg.content
                })

            logger.info(f"Sending request to Claude API with {len(messages)} messages")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    CLAUDE_API_URL,
                    headers={
                        "x-api-key": CLAUDE_API_KEY,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": agent.model,
                        "messages": messages,
                        "max_tokens": request.max_tokens or 2048,
                        "temperature": request.temperature or 0.7,
                        "stream": request.stream or False
                    }
                )
                response.raise_for_status()
                claude_response = response.json()

                # Store conversation for future context
                await self.store_conversation(messages, agent)
                
                # Format response with agent name
                response_text = f"[{agent.name}] {claude_response['content'][0]['text']}"
                
                # Convert Claude response format to match our API
                return {
                    "id": claude_response["id"],
                    "object": "chat.completion",
                    "created": claude_response["created_at"],
                    "model": agent.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                            "name": agent.name
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": claude_response.get("usage", None)
                }

        except Exception as e:
            logger.error(f"Error routing to Claude: {e}")
            raise
