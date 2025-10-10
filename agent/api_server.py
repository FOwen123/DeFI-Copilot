"""
FastAPI server for DeFi Copilot Agent with streaming support.
Compatible with OpenAI API format for easy integration with Webpilot.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
from agent import stream_agent_response, invoke_agent

app = FastAPI(
    title="DeFi Copilot API",
    description="LangGraph-powered DeFi assistant API",
    version="1.0.0"
)

# Enable CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models (OpenAI-compatible)
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    # DeFi-specific fields
    context: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "DeFi Copilot API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """
    OpenAI-compatible chat completions endpoint with streaming support.
    This endpoint is compatible with the Webpilot extension.
    
    Handles both standard OpenAI format and Webpilot's nested format.
    """
    try:
        # Handle Webpilot's nested format where messages are inside model object
        messages = None
        stream = True
        
        # Check if this is Webpilot format (messages inside model object)
        if "model" in request and isinstance(request["model"], dict):
            if "messages" in request["model"]:
                messages = request["model"]["messages"]
                stream = request.get("stream", True)
        # Standard OpenAI format
        elif "messages" in request:
            messages = request["messages"]
            stream = request.get("stream", True)
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages found in request")
        
        # Convert messages to our format, handling 'function' role from Webpilot
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Convert 'function' role to 'user' with context prefix
            if role == "function":
                formatted_messages.append({
                    "role": "user",
                    "content": f"[Webpage Context]\n{content}"
                })
            else:
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
        
        # Extract the user's query from messages
        user_messages = [msg for msg in formatted_messages if msg["role"] == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1]["content"]
        
        # Get DeFi context if provided
        context = request.get("context", {})
        
        # Get model name from request
        model_name = "gpt-4o-mini"  # default
        if isinstance(request.get("model"), str):
            model_name = request.get("model")
        elif isinstance(request.get("model"), dict):
            model_name = request.get("model", {}).get("model", "gpt-4o-mini")
        
        # If streaming is requested (default for Webpilot)
        if stream:
            async def generate():
                try:
                    async for chunk in stream_agent_response(query, context):
                        # Format as OpenAI streaming response
                        response_chunk = {
                            "id": "chatcmpl-defi",
                            "object": "chat.completion.chunk",
                            "created": 1234567890,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": chunk
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"
                    
                    # Send final chunk
                    final_chunk = {
                        "id": "chatcmpl-defi",
                        "object": "chat.completion.chunk",
                        "created": 1234567890,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "agent_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        else:
            response_text = invoke_agent(query, context)
            return {
                "id": "chatcmpl-defi",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/defi/ask")
async def defi_ask(query: str, context: Optional[Dict[str, Any]] = None):
    """
    Simplified DeFi-specific endpoint (non-OpenAI format).
    """
    try:
        response = invoke_agent(query, context)
        return {
            "query": query,
            "response": response,
            "context": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting DeFi Copilot API server on {host}:{port}")
    print(f"📊 LangSmith tracing: {'enabled' if os.getenv('LANGCHAIN_TRACING_V2') else 'disabled'}")
    print(f"\n💡 API endpoints:")
    print(f"   - Health: http://localhost:{port}/health")
    print(f"   - Chat (OpenAI compatible): http://localhost:{port}/v1/chat/completions")
    print(f"   - DeFi Ask: http://localhost:{port}/defi/ask")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=True
    )
