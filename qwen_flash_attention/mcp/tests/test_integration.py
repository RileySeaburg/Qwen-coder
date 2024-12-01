import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import Dict, Any
from ..tools import (
    ToolRegistry, ToolDefinition, ToolRequest, ToolResponse,
    ToolParameter, ToolParameterType
)
from ...rag_agent import RAGAgent, SearchResult

@pytest.fixture
async def mock_rag_agent():
    with patch('qwen_flash_attention.rag_agent.RAGAgent') as mock:
        agent = AsyncMock(spec=RAGAgent)
        agent.search_similar.return_value = [
            {
                "text": "Test document",
                "metadata": {"source": "test"},
                "score": 0.9
            }
        ]
        agent.add_documents.return_value = ["doc_id"]
        yield agent

@pytest.fixture
async def tool_registry(mock_rag_agent):
    registry = ToolRegistry()
    registry.rag_agent = mock_rag_agent
    await registry.initialize()
    return registry

class TestFileOperations:
    """Test file operation tools with mocked filesystem."""
    
    @pytest.fixture
    def mock_filesystem(self):
        with patch('builtins.open') as mock_open:
            mock_file = AsyncMock()
            mock_open.return_value.__enter__.return_value = mock_file
            yield mock_file

    async def test_read_file(self, tool_registry, mock_filesystem):
        """Test file reading with mocked file."""
        mock_filesystem.read.return_value = "Test content"
        
        response = await tool_registry.execute_tool(ToolRequest(
            tool="readFile",
            parameters={"path": "test.txt"}
        ))
        
        assert response.error is None
        assert response.result == "Test content"
        mock_filesystem.read.assert_called_once()

    async def test_write_file(self, tool_registry, mock_filesystem):
        """Test file writing with mocked file."""
        response = await tool_registry.execute_tool(ToolRequest(
            tool="writeFile",
            parameters={
                "path": "test.txt",
                "content": "Test content"
            }
        ))
        
        assert response.error is None
        assert response.result is True
        mock_filesystem.write.assert_called_once_with("Test content")

class TestKnowledgeOperations:
    """Test knowledge base operations with mocked RAG agent."""

    async def test_search_knowledge(self, tool_registry, mock_rag_agent):
        """Test knowledge search."""
        response = await tool_registry.execute_tool(ToolRequest(
            tool="searchKnowledge",
            parameters={
                "query": "test query",
                "num_results": 1
            }
        ))
        
        assert response.error is None
        assert len(response.result) == 1
        assert response.result[0]["text"] == "Test document"
        mock_rag_agent.search_similar.assert_called_once_with(
            query="test query",
            num_results=1
        )

    async def test_add_knowledge(self, tool_registry, mock_rag_agent):
        """Test knowledge addition."""
        response = await tool_registry.execute_tool(ToolRequest(
            tool="addKnowledge",
            parameters={
                "text": "Test document",
                "metadata": {"source": "test"}
            }
        ))
        
        assert response.error is None
        assert response.result is True
        mock_rag_agent.add_documents.assert_called_once()

class TestCommandExecution:
    """Test command execution with mocked subprocess."""
    
    @pytest.fixture
    def mock_subprocess(self):
        with patch('subprocess.Popen') as mock_popen:
            process = AsyncMock()
            process.communicate.return_value = ("stdout", "stderr")
            process.returncode = 0
            mock_popen.return_value = process
            yield process

    async def test_execute_command(self, tool_registry, mock_subprocess):
        """Test command execution."""
        response = await tool_registry.execute_tool(ToolRequest(
            tool="executeCommand",
            parameters={"command": "test command"}
        ))
        
        assert response.error is None
        assert response.result["stdout"] == "stdout"
        assert response.result["stderr"] == "stderr"
        assert response.result["returncode"] == 0
        mock_subprocess.communicate.assert_called_once()

class TestErrorHandling:
    """Test error handling in integration scenarios."""

    async def test_file_not_found(self, tool_registry):
        """Test handling of file not found error."""
        response = await tool_registry.execute_tool(ToolRequest(
            tool="readFile",
            parameters={"path": "nonexistent.txt"}
        ))
        
        assert response.error is not None
        assert "No such file" in response.error

    async def test_rag_error(self, tool_registry, mock_rag_agent):
        """Test handling of RAG agent errors."""
        mock_rag_agent.search_similar.side_effect = Exception("RAG error")
        
        response = await tool_registry.execute_tool(ToolRequest(
            tool="searchKnowledge",
            parameters={"query": "test"}
        ))
        
        assert response.error is not None
        assert "RAG error" in response.error

    async def test_command_error(self, tool_registry, mock_subprocess):
        """Test handling of command execution errors."""
        mock_subprocess.communicate.side_effect = Exception("Command error")
        
        response = await tool_registry.execute_tool(ToolRequest(
            tool="executeCommand",
            parameters={"command": "test"}
        ))
        
        assert response.error is not None
        assert "Command error" in response.error

class TestConcurrency:
    """Test concurrent operations with external systems."""

    async def test_concurrent_rag_operations(self, tool_registry, mock_rag_agent):
        """Test concurrent RAG operations."""
        tasks = []
        for i in range(5):
            request = ToolRequest(
                tool="searchKnowledge",
                parameters={"query": f"test{i}"}
            )
            tasks.append(tool_registry.execute_tool(request))

        responses = await asyncio.gather(*tasks)
        assert all(r.error is None for r in responses)
        assert mock_rag_agent.search_similar.call_count == 5

    async def test_concurrent_file_operations(self, tool_registry, mock_filesystem):
        """Test concurrent file operations."""
        tasks = []
        for i in range(5):
            request = ToolRequest(
                tool="writeFile",
                parameters={
                    "path": f"test{i}.txt",
                    "content": f"content{i}"
                }
            )
            tasks.append(tool_registry.execute_tool(request))

        responses = await asyncio.gather(*tasks)
        assert all(r.error is None for r in responses)
        assert mock_filesystem.write.call_count == 5
