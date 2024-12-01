# Combined API Specification

## Base URL
`http://localhost:8000`

## Endpoints

### Health Check
```
GET /health
```
Returns server status and model information.

Response:
```json
{
  "status": "healthy",
  "gpu": {
    "name": "string",
    "memory_allocated": "number",
    "memory_cached": "number"
  },
  "models": ["string"],
  "anthropic_client": "boolean"
}
```

### Chat Completion
```
POST /v1/chat/completions
```
Standard chat completion endpoint for Qwen model.

Request:
```json
{
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ],
  "model": "string",
  "temperature": "number",
  "max_tokens": "number",
  "stream": "boolean"
}
```

Response:
```json
{
  "id": "string",
  "object": "chat.completion",
  "created": "number",
  "model": "string",
  "choices": [
    {
      "index": "number",
      "message": {
        "role": "string",
        "content": "string"
      },
      "finish_reason": "string"
    }
  ],
  "usage": {
    "prompt_tokens": "number",
    "completion_tokens": "number",
    "total_tokens": "number"
  }
}
```

### Agent Chat
```
POST /v1/agent/chat
```
AutoGen-powered chat endpoint with support for file editing and RAG.

Request:
```json
{
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ],
  "config": {
    "name": "string",
    "role": "string",
    "anthropic_api_key": "string (optional)"
  },
  "max_tokens": "number (optional, default: 512)",
  "temperature": "number (optional, default: 0.2)"
}
```

Response:
```json
{
  "role": "string",
  "content": "string",
  "provider": "string (qwen | anthropic)"
}
```

### Create Agent Group
```
POST /v1/groups/create
```
Create a new agent team configuration.

Request:
```json
{
  "agents": [
    {
      "name": "string",
      "role": "string",
      "model": "string",
      "systemPrompt": "string"
    }
  ],
  "teamType": "string (round_robin)"
}
```

Response:
```json
{
  "group_id": "string",
  "config": {
    "agents": [
      {
        "name": "string",
        "role": "string",
        "model": "string",
        "systemPrompt": "string"
      }
    ],
    "teamType": "string"
  }
}
```

### Group Chat
```
POST /v1/groups/{group_id}/chat
```
Chat with a specific agent group.

Request:
```json
{
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ],
  "model": "string",
  "temperature": "number (optional)",
  "max_tokens": "number (optional)"
}
```

Response:
```json
{
  "id": "string",
  "object": "chat.completion",
  "created": "number",
  "model": "string",
  "choices": [
    {
      "index": "number",
      "message": {
        "role": "string",
        "content": "string"
      },
      "finish_reason": "string"
    }
  ],
  "usage": {
    "prompt_tokens": "number",
    "completion_tokens": "number",
    "total_tokens": "number"
  }
}
```

## WebSocket Events

### Agent Messages
```
WebSocket /ws
```
Receive real-time messages from agents.

Message Format:
```json
{
  "type": "agent_message",
  "agent": {
    "name": "string",
    "role": "string"
  },
  "content": "string",
  "timestamp": "number"
}
```

## Models

Available models:
- `qwen2.5-coder-1.5b-instruct-gguf`: Qwen 1.5B model
- `qwen2.5-coder-3b-instruct-gguf`: Qwen 3B model
- `qwen2.5-coder-7b-instruct-gguf`: Qwen 7B model
- `qwen2.5-coder-14b-instruct-gguf`: Qwen 14B model
- `claude-3-sonnet-20240229`: Claude model (requires API key)

## Error Responses

All endpoints may return the following error responses:

```json
{
  "detail": "string"
}
```

Status codes:
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error
