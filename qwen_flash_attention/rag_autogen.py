import sys
import os
import json
import re
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autogen/python"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autogen/python/packages/autogen-core/src"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autogen/python/packages/autogen-magentic-one/src"))

from typing import List, Dict, Any, Optional, cast
import logging
import time
from fastapi import Request
from .rag_provider import RAGProvider
from .schemas import Message, Agent
from autogen_core.base import CancellationToken
from autogen_core.components.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
    LLMMessage
)

logger = logging.getLogger(__name__)

class RAGAutogenTeam:
    """Team using AutoGen agents with RAG and web surfing capabilities."""
    
    def __init__(
        self,
        config_list: List[Dict[str, Any]],
        mongodb_config: Dict[str, Any],
        request: Optional[Request] = None
    ):
        self.config_list = config_list
        self.assistant: Optional[Agent] = None
        self.user_proxy: Optional[Agent] = None
        self.rag_provider = RAGProvider(mongodb_config, request)
        self.request = request
        self.initialized = False

    async def initialize(self):
        """Initialize the team."""
        if not self.initialized:
            try:
                logger.info("Initializing team...")

                # Initialize RAG provider
                await self.rag_provider.initialize()

                # Initialize assistant agent
                self.assistant = Agent(
                    name="assistant",
                    role="assistant",
                    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    systemPrompt="You are a helpful assistant. Use the provided context to inform your responses."
                )

                # Initialize user proxy agent
                self.user_proxy = Agent(
                    name="user_proxy",
                    role="user",
                    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    systemPrompt="You are a user proxy that helps coordinate tasks."
                )

                # Initialize web surfer if needed
                if any("web_researcher" in agent.get("role", "") for agent in self.config_list):
                    logger.info("Initializing web surfer...")
                    # Import here to avoid circular imports
                    from autogen_magentic_one.agents.multimodal_web_surfer import MultimodalWebSurfer
                    self.web_surfer = MultimodalWebSurfer(
                        description="A web researcher that can browse websites and extract information."
                    )
                    if self.request and hasattr(self.request.app.state, "qwen_model"):
                        await self.web_surfer.init(
                            model_client=self.request.app.state.qwen_model,
                            headless=True,
                            browser_channel="chromium",
                            downloads_folder="downloads",
                            debug_dir="debug",
                            to_save_screenshots=True
                        )
                        logger.info("Web surfer initialized successfully")

                self.initialized = True
                logger.info("Team initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing team: {e}")
                raise

    def _extract_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract function call from JSON format."""
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return None

    async def process_task(self, task: str) -> Dict[str, Any]:
        """Process a task using the team."""
        if not self.initialized:
            await self.initialize()

        try:
            logger.info(f"Processing task: {task}")

            if not self.assistant:
                raise RuntimeError("Team not properly initialized")

            if not self.request or not hasattr(self.request.app.state, "qwen_model"):
                raise RuntimeError("Qwen model not available")

            qwen_model = self.request.app.state.qwen_model

            # Create initial message
            initial_message = Message(
                role="user",
                content=task,
                name="user"
            )

            # Enhance message with context
            enhanced_messages = await self.rag_provider.enhance_messages([initial_message])
            
            # Format messages for Qwen model
            messages = []
            for msg in enhanced_messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name
                })

            # Generate response using Qwen model
            response_text = await qwen_model.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                tools=["browser_action"] if hasattr(self, 'web_surfer') else None
            )

            # Extract and handle function calls
            if hasattr(self, 'web_surfer'):
                function_call = self._extract_function_call(response_text)
                if function_call and function_call.get("name") == "browser_action":
                    logger.info(f"Executing browser action: {function_call}")
                    try:
                        # Convert function call to AutoGen format
                        action = function_call["arguments"]["action"]
                        if action == "launch":
                            web_response = await self.web_surfer._generate_reply(CancellationToken())
                            web_content = web_response[1][0] if isinstance(web_response[1], list) else web_response[1]
                            response_text = web_content
                    except Exception as e:
                        logger.error(f"Error executing browser action: {e}")

            # Format response
            response = {
                "role": "assistant",
                "content": response_text,
                "name": self.assistant.name
            }

            logger.info("Task processed successfully")
            return response
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            raise

    async def add_to_knowledge_base(self, documents: List[Dict[str, Any]], source: str) -> List[str]:
        """Add documents to the knowledge base."""
        if not self.initialized:
            await self.initialize()
        return await self.rag_provider.add_to_knowledge_base(documents, source)

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'web_surfer') and self.web_surfer and self.web_surfer._playwright:
                import asyncio
                asyncio.create_task(self.web_surfer._playwright.stop())
            del self.rag_provider
            logger.info("Team cleaned up")
        except:
            pass
