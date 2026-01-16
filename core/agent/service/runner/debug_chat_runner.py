import json
from typing import Any, AsyncGenerator

# Use unified common package import module
from common.otlp.log_trace.node_log import Data as NodeData
from common.otlp.log_trace.node_log import NodeLog as Node
from common.otlp.log_trace.node_trace_log import NodeTraceLog as NodeTrace
from common.otlp.trace.span import Span
from pydantic import BaseModel, ConfigDict, Field

from agent.api.schemas.agent_response import AgentResponse
from agent.api.schemas.completion_chunk import (
    ReasonChatCompletionChunk,
    ReasonChoiceDeltaToolCall,
    ReasonChoiceDeltaToolCallFunction,
)
from agent.engine.bot.runner import BotCotStep, BotRunner, BotAgentResponse
from agent.api.schemas_v2.bot_debug_chat_response import (
    BotDebugChatCompletionChunk,
    BotDebugChatChoice,
    BotDebugChatChoiceDelta,
    BotDebugChatChoiceDeltaToolCall,
    BotDebugChatChoiceDeltaToolCallFunction,
    BotDebugChatChoiceDeltaToolCallResponse,
    BotDebugChatChoiceDeltaToolCallResponseResponse,
)


class UpdatedNode(Node):
    node_name: str = Field(default="")


class DebugChatRunner(BaseModel):
    """Bot Agent runner"""

    bot_runner: BotRunner

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def run(
        self, span: Span, node_trace: NodeTrace
    ) -> AsyncGenerator[BotDebugChatCompletionChunk, None]:
        """Execute"""

        async for message in self.bot_runner.run(span, node_trace):
            yield await self.convert_message(message, span=span, node_trace=node_trace)

    async def convert_message(
        self, message: BotAgentResponse, span: Span, node_trace: NodeTrace
    ) -> BotDebugChatCompletionChunk:
        """Convert BotAgentResponse to return frame"""

        chunk = BotDebugChatCompletionChunk(
            id=str(span.sid),
            choices=[BotDebugChatChoice(index=0, delta=BotDebugChatChoiceDelta())],
            created=message.created,
            usage=message.usage,
        )

        if message.typ == "tool_call":
            await self._handle_tool_call(chunk, message, span, node_trace)
        elif message.typ == "tool_call_result":
            self._handle_tool_call_result(chunk, message)

        if message.typ == "reasoning_content":
            self._handle_reasoning_content(chunk, message)
        elif message.typ == "content":
            self._handle_content(chunk, message)

        return chunk

    def _handle_reasoning_content(
        self, chunk: ReasonChatCompletionChunk, message: AgentResponse
    ) -> None:
        """Handle reasoning content"""
        if isinstance(message.content, str):
            chunk.choices[0].delta.reasoning_content = message.content

    def _handle_content(
        self, chunk: ReasonChatCompletionChunk, message: AgentResponse
    ) -> None:
        """Handle normal content"""
        if isinstance(message.content, str):
            chunk.choices[0].delta.content = message.content

    async def _handle_tool_call(
        self,
        chunk: BotDebugChatCompletionChunk,
        message: BotAgentResponse,
        span: Span,
        node_trace: NodeTrace,
    ) -> None:
        """Handle CoT step"""

        content = message.content
        try:
            chunk.choices[0].delta.tool_calls = [
                BotDebugChatChoiceDeltaToolCall(
                    index=0,
                    type=tool_call["plugin"].typ,
                    function=BotDebugChatChoiceDeltaToolCallFunction(
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                    ),
                )
                for tool_call in content
            ]
        except Exception as e:
            raise e

        # self._handle_plugin_trace(content, span, node_trace)

    def _handle_plugin_trace(
        self,
        content: BotCotStep,
        span: Span,
        node_trace: NodeTrace,
    ) -> None:
        """Handle plugin trace"""
        called_plugin = getattr(content, "plugin", None)
        if not (
            called_plugin is not None
            and hasattr(called_plugin, "run_result")
            and called_plugin.run_result is not None
        ):
            return

        run_result = called_plugin.run_result
        start_time = getattr(run_result, "start_time", 0)
        end_time = getattr(run_result, "end_time", 0)
        thought = getattr(content, "thought", "") if hasattr(content, "thought") else ""

        node_trace.trace.append(
            UpdatedNode(
                id=getattr(run_result, "sid", ""),
                sid=span.sid,
                node_id=self._determine_node_id(called_plugin),
                node_name=getattr(called_plugin, "name", ""),
                node_type=getattr(called_plugin, "typ", ""),
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                running_status=not bool(getattr(run_result, "code", 0)),
                logs=[thought] if thought else [],
                data=NodeData(
                    input=content.action_input if content.action_input else {},
                    output=content.action_output if content.action_output else {},
                ),
            )
        )

    def _determine_node_id(self, called_plugin: Any) -> str:
        """Determine node ID"""
        if not hasattr(called_plugin, "typ"):
            return ""

        plugin_type = called_plugin.typ
        node_id = ""

        if plugin_type == "workflow" and hasattr(called_plugin, "flow_id"):
            node_id = called_plugin.flow_id
        elif plugin_type == "mcp":
            if hasattr(called_plugin, "server_id") and called_plugin.server_id:
                node_id = called_plugin.server_id
            elif hasattr(called_plugin, "server_url") and called_plugin.server_url:
                node_id = called_plugin.server_url
        elif hasattr(called_plugin, "tool_id"):
            node_id = called_plugin.tool_id

        return node_id

    def _handle_log(
        self, chunk: ReasonChatCompletionChunk, message: AgentResponse
    ) -> None:
        """Handle log message"""
        chunk.object = "chat.completion.log"
        if isinstance(message.content, str):
            chunk.logs.append(message.content)

    def _handle_knowledge_metadata(
        self, chunk: ReasonChatCompletionChunk, message: AgentResponse
    ) -> None:
        """Handle knowledge metadata"""
        chunk.choices[0].delta.tool_calls = [
            ReasonChoiceDeltaToolCall(
                index=0,
                type="knowledge",
                reason="",
                function=ReasonChoiceDeltaToolCallFunction(
                    name="knowledge",
                    arguments=json.dumps(
                        {"query": getattr(self.chat_runner, "question", "")},
                        ensure_ascii=False,
                    ),
                    response=json.dumps(
                        {"metadata_list": message.content}, ensure_ascii=False
                    ),
                ),
            )
        ]

    def _handle_tool_call_result(
        self, chunk: BotDebugChatCompletionChunk, message: BotAgentResponse
    ) -> None:
        """Handle tool call execution results."""
        results = message.content or []
        if not isinstance(results, list):
            results = [results]

        chunk.choices[0].delta.tool_call_responses = [
            BotDebugChatChoiceDeltaToolCallResponse(
                id=str(r.get("id", "")) if isinstance(r, dict) else "",
                response_type=str(r.get("plugin_type", r.get("response_type", "")))
                if isinstance(r, dict)
                else "",
                stream=False,
                chunks=[],
                response=BotDebugChatChoiceDeltaToolCallResponseResponse(
                    content_type="json",
                    content=(
                        r.get("response", r)
                        if isinstance(r, dict)
                        else r
                    ),
                ),
            )
            for r in results
        ]
