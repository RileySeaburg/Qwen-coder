import pytest
import asyncio
import os
import sys
from typing import AsyncGenerator
from qwen_flash_attention.mcp.tools import ToolRegistry, default_registry
from qwen_flash_attention.rag_agent import RAGAgent

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Configure pytest
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "system: mark test as system test"
    )

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def mongodb():
    """Start MongoDB for testing."""
    import motor.motor_asyncio
    client = motor.motor_asyncio.AsyncIOMotorClient(
        "mongodb://localhost:27017",
        serverSelectionTimeoutMS=5000
    )
    await client.admin.command('ping')
    yield client
    client.close()

@pytest.fixture(scope="session")
async def rag_agent(mongodb):
    """Create RAG agent for testing."""
    agent = RAGAgent(
        mongodb_url="mongodb://localhost:27017",
        database_name="test_vectors",
        collection_name="test_knowledge"
    )
    await agent.initialize()
    yield agent
    # Cleanup test database
    await mongodb.drop_database("test_vectors")

@pytest.fixture(scope="session")
async def tool_registry(rag_agent):
    """Create tool registry for testing."""
    registry = ToolRegistry()
    registry.rag_agent = rag_agent
    await registry.initialize()
    yield registry

@pytest.fixture(autouse=True)
async def cleanup_files():
    """Clean up test files after each test."""
    yield
    # Remove test files
    test_files = [
        "test.py",
        "test.txt",
        "test_output.txt"
    ]
    for file in test_files:
        try:
            os.remove(file)
        except OSError:
            pass
