import logging
from typing import List, Dict, Any, Optional, Union, cast
from autogen_schemas import ChatCompletionRequest, Agent, Message
import json
import re

logger = logging.getLogger(__name__)

class MetaModel:
    """Meta model for AutoGen task routing and agent coordination."""
    
    def __init__(self):
        self.task_patterns = {
            "code_generation": [
                r"write|create|implement|generate.*code",
                r"function|class|method|program",
                r"coding|programming|development"
            ],
            "code_review": [
                r"review|analyze|check.*code",
                r"find.*bugs|issues|problems",
                r"improve|optimize|refactor"
            ],
            "system_operation": [
                r"run|execute|install|setup",
                r"configure|deploy|launch",
                r"system|server|service"
            ],
            "file_operation": [
                r"read|write|modify|update.*file",
                r"create.*file|directory",
                r"file.*operation|management"
            ],
            "planning": [
                r"plan|design|architect",
                r"strategy|approach|solution",
                r"organize|structure|layout"
            ],
            "research": [
                r"research|investigate|explore",
                r"find.*information|details",
                r"learn|understand|study"
            ]
        }

    def analyze_task(self, messages: List[Message]) -> Dict[str, Any]:
        """Analyze task from conversation history."""
        # Get the latest user message
        user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
        if not user_message:
            return {"type": "unknown", "confidence": 0.0}

        content = user_message.content.lower()
        task_scores = {}

        # Score each task type based on pattern matches
        for task_type, patterns in self.task_patterns.items():
            score = 0
            matches = 0
            for pattern in patterns:
                if re.search(pattern, content):
                    matches += 1
                    score += 1

            if matches > 0:
                confidence = score / len(patterns)
                task_scores[task_type] = confidence

        if not task_scores:
            return {"type": "general", "confidence": 1.0}

        # Get task type with highest confidence
        best_task = max(task_scores.items(), key=lambda x: x[1])
        return {
            "type": best_task[0],
            "confidence": best_task[1]
        }

    def select_agent(self, task_analysis: Dict[str, Any], agents: List[Agent]) -> Agent:
        """Select the most appropriate agent for the task."""
        task_type = task_analysis["type"]
        
        # Define agent capabilities
        agent_capabilities = {
            "code_generation": ["qwen", "coder"],
            "code_review": ["qwen", "coder", "reviewer"],
            "system_operation": ["claude"],
            "file_operation": ["claude"],
            "planning": ["claude", "architect"],
            "research": ["claude", "researcher"],
            "general": ["qwen", "claude"]
        }

        # Score each agent based on capabilities
        agent_scores = {}
        capabilities = agent_capabilities.get(task_type, ["qwen"])
        
        for agent in agents:
            score = 0
            # Check agent name and role against capabilities
            agent_attrs = [
                agent.name.lower(),
                agent.role.lower(),
                agent.model.lower()
            ]
            for capability in capabilities:
                if any(capability in attr for attr in agent_attrs):
                    score += 1
            agent_scores[agent] = score

        # Select agent with highest score
        selected_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected agent {selected_agent.name} for task type {task_type}")
        return selected_agent

    def format_prompt(self, task_analysis: Dict[str, Any], agent: Union[Agent, str], messages: List[Message]) -> List[Message]:
        """Format prompt for selected agent."""
        task_type = task_analysis["type"]
        
        # Add system message with task-specific instructions
        system_instructions = {
            "code_generation": "Focus on writing clean, efficient, and well-documented code.",
            "code_review": "Analyze code for bugs, performance issues, and best practices.",
            "system_operation": "Execute system commands carefully and provide clear feedback.",
            "file_operation": "Handle files with care, always verify operations.",
            "planning": "Provide structured, detailed plans and consider edge cases.",
            "research": "Focus on gathering accurate, relevant information and cite sources.",
            "general": "Provide helpful, accurate responses."
        }

        # Handle agent being either Agent object or string
        agent_name = agent.name if isinstance(agent, Agent) else str(agent)
        agent_prompt = agent.systemPrompt if isinstance(agent, Agent) else "Assistant"

        formatted_messages = [
            Message(
                role="system",
                content=f"You are {agent_name}, {agent_prompt}\n\n"
                        f"Task Type: {task_type}\n"
                        f"{system_instructions.get(task_type, '')}",
                name="system"
            )
        ]

        # Add conversation history
        formatted_messages.extend(messages)

        return formatted_messages

    def analyze_response(self, response: str, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent response for quality and completeness."""
        task_type = task_analysis["type"]
        
        # Define success criteria for each task type
        success_patterns = {
            "code_generation": [
                r"```.*```",  # Code blocks
                r"function|class|def|import",  # Code elements
                r"explanation|usage|example"  # Documentation
            ],
            "code_review": [
                r"issue|bug|problem",  # Found issues
                r"suggestion|recommendation",  # Improvements
                r"best practice|pattern"  # Standards
            ],
            "system_operation": [
                r"executed|completed|done",  # Operation status
                r"output|result|error",  # Operation result
                r"command|instruction"  # Operation details
            ],
            "file_operation": [
                r"file|directory",  # File operations
                r"created|modified|updated",  # Operation status
                r"content|data"  # File details
            ],
            "planning": [
                r"step|phase|stage",  # Plan structure
                r"approach|strategy",  # Methodology
                r"consideration|requirement"  # Analysis
            ],
            "research": [
                r"found|discovered",  # Findings
                r"source|reference",  # Citations
                r"information|detail"  # Content
            ]
        }

        patterns = success_patterns.get(task_type, [r"response|answer"])
        matches = sum(1 for pattern in patterns if re.search(pattern, response.lower()))
        quality_score = matches / len(patterns) if patterns else 1.0

        return {
            "quality": quality_score,
            "complete": quality_score > 0.7,
            "needs_followup": quality_score < 0.5
        }

    async def route_task(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Route task to appropriate agent and format request."""
        # Analyze task
        task_analysis = self.analyze_task(request.messages)
        logger.info(f"Task analysis: {json.dumps(task_analysis, indent=2)}")

        # Select agent
        if request.agent_config:
            selected_agent = self.select_agent(task_analysis, request.agent_config.agents)
        else:
            # Use default agent if no team config
            selected_agent = request.messages[-1].name if request.messages[-1].name else "assistant"

        # Format prompt
        formatted_messages = self.format_prompt(task_analysis, selected_agent, request.messages)

        return {
            "task_analysis": task_analysis,
            "selected_agent": selected_agent,
            "formatted_messages": formatted_messages
        }
