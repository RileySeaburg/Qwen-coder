# MCP Tests

Test suite for Model Context Protocol implementation.

## Test Structure

```
tests/
├── conftest.py          # Test configuration and fixtures
├── pytest.ini          # Pytest configuration
├── run_tests.py        # Test runner script
├── test_functional.py  # Functional tests
├── test_integration.py # Integration tests
└── test_system.py      # System/VSCode integration tests
```

## Prerequisites

1. MongoDB running locally:
```bash
mongod --dbpath ./data
```

2. Install test dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests
```bash
python -m qwen_flash_attention.mcp.tests.run_tests
```

### Run with Coverage
```bash
python -m qwen_flash_attention.mcp.tests.run_tests --coverage
```

### Run Specific Test Types
```bash
# Functional tests only
pytest qwen_flash_attention/mcp/tests/test_functional.py

# Integration tests only
pytest qwen_flash_attention/mcp/tests/test_integration.py

# System tests only
pytest qwen_flash_attention/mcp/tests/test_system.py
```

## Test Categories

### Functional Tests
- Tool registration and validation
- Parameter validation
- Error handling
- Concurrent operations

### Integration Tests
- File operations with mocked filesystem
- Knowledge base operations with RAG
- Command execution with subprocess
- Error propagation

### System Tests
- VSCode extension integration
- Full chat completion flow
- File editing workflow
- Knowledge base integration
- Command execution
- Error handling
- Concurrent operations

## Environment Variables

The test suite uses these environment variables (configured in pytest.ini):

- `MCP_MODEL_SERVER`: Model server URL (default: http://localhost:8001)
- `MCP_CONTEXT_SERVER`: Context server URL (default: http://localhost:8000)
- `MONGODB_URL`: MongoDB connection URL (default: mongodb://localhost:27017)
- `TEST_DATABASE`: Test database name (default: test_vectors)
- `TEST_COLLECTION`: Test collection name (default: test_knowledge)

## Test Coverage

Coverage reports are generated when running with `--coverage`:

- Terminal report showing missing lines
- HTML report in `htmlcov/` directory

## Fixtures

Key fixtures available in tests:

- `tool_registry`: Initialized tool registry
- `rag_agent`: Configured RAG agent
- `mongodb`: MongoDB test connection
- `test_data`: Sample test data
- `mock_responses`: Mock model responses
- `cleanup_files`: Auto-cleanup of test files

## Adding New Tests

1. Choose appropriate test category:
   - `test_functional.py` for tool functionality
   - `test_integration.py` for external system integration
   - `test_system.py` for VSCode/end-to-end tests

2. Use relevant markers:
   ```python
   @pytest.mark.functional
   @pytest.mark.integration
   @pytest.mark.system
   ```

3. Use provided fixtures:
   ```python
   async def test_example(tool_registry, test_data):
       # Test implementation
   ```

4. Follow test patterns:
   - Arrange: Set up test conditions
   - Act: Execute functionality
   - Assert: Verify results
   - Cleanup: Use cleanup_files fixture

## Debugging

1. Enable debug logging:
   ```bash
   pytest --log-cli-level=DEBUG
   ```

2. Use VSCode debugger:
   - Set breakpoints in test files
   - Run "Python: Debug Tests" configuration

3. Check server logs:
   - Model server: `logs/model_server.log`
   - Context server: `logs/context_server.log`

## Common Issues

1. MongoDB Connection:
   - Ensure MongoDB is running
   - Check connection URL
   - Verify permissions

2. Server Startup:
   - Check ports 8000/8001 are free
   - Verify server logs
   - Check process permissions

3. Test Failures:
   - Check environment variables
   - Verify test data
   - Review server logs
   - Check cleanup state
