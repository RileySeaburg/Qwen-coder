import pytest
import asyncio
from typing import Dict, Any
from ..tools import (
    ToolRegistry, ToolDefinition, ToolRequest, ToolResponse,
    ToolParameter, ToolParameterType
)

@pytest.fixture
async def tool_registry():
    registry = ToolRegistry()
    await registry.initialize()
    return registry

@pytest.fixture
def sample_tool_definition():
    return ToolDefinition(
        name="testTool",
        description="Test tool for unit tests",
        parameters=[
            ToolParameter(
                name="param1",
                type=ToolParameterType.STRING,
                description="Test parameter"
            ),
            ToolParameter(
                name="param2",
                type=ToolParameterType.NUMBER,
                description="Optional parameter",
                required=False
            )
        ],
        returns={"type": "string", "description": "Test result"}
    )

class TestToolFunctionality:
    async def test_valid_input(self, tool_registry: ToolRegistry, sample_tool_definition: ToolDefinition):
        """Test tool execution with valid input."""
        async def test_handler(param1: str, param2: float = 1.0) -> str:
            return f"Result: {param1}, {param2}"

        tool_registry.register_tool(sample_tool_definition, test_handler)
        
        # Test with required parameter only
        response = await tool_registry.execute_tool(ToolRequest(
            tool="testTool",
            parameters={"param1": "test"}
        ))
        assert response.error is None
        assert response.result == "Result: test, 1.0"

        # Test with all parameters
        response = await tool_registry.execute_tool(ToolRequest(
            tool="testTool",
            parameters={"param1": "test", "param2": 2.0}
        ))
        assert response.error is None
        assert response.result == "Result: test, 2.0"

    async def test_invalid_input(self, tool_registry: ToolRegistry, sample_tool_definition: ToolDefinition):
        """Test tool execution with invalid input."""
        async def test_handler(param1: str, param2: float = 1.0) -> str:
            return f"Result: {param1}, {param2}"

        tool_registry.register_tool(sample_tool_definition, test_handler)
        
        # Test missing required parameter
        response = await tool_registry.execute_tool(ToolRequest(
            tool="testTool",
            parameters={}
        ))
        assert response.error is not None
        assert "Missing required parameter" in response.error

        # Test wrong parameter type
        response = await tool_registry.execute_tool(ToolRequest(
            tool="testTool",
            parameters={"param1": 123}  # Should be string
        ))
        assert response.error is not None
        assert "Invalid parameter type" in response.error

    async def test_parameter_validation(self, tool_registry: ToolRegistry):
        """Test parameter validation for different types."""
        tool_def = ToolDefinition(
            name="validationTool",
            description="Tool for testing parameter validation",
            parameters=[
                ToolParameter(
                    name="string_param",
                    type=ToolParameterType.STRING,
                    description="String parameter"
                ),
                ToolParameter(
                    name="number_param",
                    type=ToolParameterType.NUMBER,
                    description="Number parameter"
                ),
                ToolParameter(
                    name="boolean_param",
                    type=ToolParameterType.BOOLEAN,
                    description="Boolean parameter"
                ),
                ToolParameter(
                    name="array_param",
                    type=ToolParameterType.ARRAY,
                    description="Array parameter",
                    items={"type": "string"}
                ),
                ToolParameter(
                    name="object_param",
                    type=ToolParameterType.OBJECT,
                    description="Object parameter",
                    properties={"key": "string"}
                )
            ],
            returns={"type": "object", "description": "Validation result"}
        )

        async def validation_handler(**params) -> Dict[str, Any]:
            return params

        tool_registry.register_tool(tool_def, validation_handler)

        # Test valid parameters
        response = await tool_registry.execute_tool(ToolRequest(
            tool="validationTool",
            parameters={
                "string_param": "test",
                "number_param": 123,
                "boolean_param": True,
                "array_param": ["a", "b", "c"],
                "object_param": {"key": "value"}
            }
        ))
        assert response.error is None
        assert response.result["string_param"] == "test"
        assert response.result["number_param"] == 123
        assert response.result["boolean_param"] is True
        assert response.result["array_param"] == ["a", "b", "c"]
        assert response.result["object_param"] == {"key": "value"}

    async def test_error_propagation(self, tool_registry: ToolRegistry, sample_tool_definition: ToolDefinition):
        """Test error propagation from tool handler."""
        async def error_handler(**params) -> str:
            raise ValueError("Test error")

        tool_registry.register_tool(sample_tool_definition, error_handler)
        
        response = await tool_registry.execute_tool(ToolRequest(
            tool="testTool",
            parameters={"param1": "test"}
        ))
        assert response.error is not None
        assert "Test error" in response.error

    async def test_concurrent_execution(self, tool_registry: ToolRegistry, sample_tool_definition: ToolDefinition):
        """Test concurrent tool execution."""
        async def slow_handler(param1: str) -> str:
            await asyncio.sleep(0.1)
            return f"Result: {param1}"

        tool_registry.register_tool(sample_tool_definition, slow_handler)
        
        # Execute multiple tools concurrently
        tasks = []
        for i in range(5):
            request = ToolRequest(
                tool="testTool",
                parameters={"param1": f"test{i}"}
            )
            tasks.append(tool_registry.execute_tool(request))

        responses = await asyncio.gather(*tasks)
        assert all(r.error is None for r in responses)
        assert len(set(r.result for r in responses)) == 5  # All results should be unique
