import time
from typing import List, Literal, Optional, Any

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import Field, BaseModel

def cur_timestamp() -> int:
    return int(time.time() * 1000)

class BotDebugChatChoiceDeltaToolCallFunction(ChoiceDeltaToolCallFunction):
    arguments: dict


class BotDebugChatChoiceDeltaToolCall(ChoiceDeltaToolCall):
    function: Optional[BotDebugChatChoiceDeltaToolCallFunction] = None
    type: Optional[Literal["workflow", "link", "knowledge", "mcp"]] = None  # type: ignore[assignment]
    reason: str = Field(default="")


class BotDebugChatChoiceDeltaToolCallResponseResponse(BaseModel):
    content_type: str
    content: Any


class BotDebugChatChoiceDeltaToolCallResponse(BaseModel):
    id: str
    response_type: str
    stream: bool
    chunks: list = Field(default_factory=List)
    response: BotDebugChatChoiceDeltaToolCallResponseResponse


class BotDebugChatChoiceDelta(ChoiceDelta):
    role: Optional[Literal["assistant"]] = Field(default="assistant")
    content: str = Field(default="")
    reasoning_content: str = Field(default="")
    tool_calls: List[BotDebugChatChoiceDeltaToolCall] = Field(default_factory=list)  # type: ignore[assignment]
    tool_call_responses: List[BotDebugChatChoiceDeltaToolCallResponse] = Field(
        default_factory=list
    )


class BotDebugChatChoice(Choice):
    delta: BotDebugChatChoiceDelta


class BotDebugChatCompletionChunk(ChatCompletionChunk):
    code: int = Field(default=0)
    message: str = Field(default="success")
    id: str = Field(default="")
    created: int = Field(default_factory=cur_timestamp)
    choices: List[BotDebugChatChoice] = Field(default_factory=list)
    object: Literal["chat.completion.chunk"] = Field(default="chat.completion.chunk")
    model: str = Field(default="")
