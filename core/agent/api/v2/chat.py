import json
import time
from typing import Annotated, Any, AsyncGenerator, cast, List

from agent.api.schemas_v2.bot_debug_chat_response import BotDebugChatCompletionChunk
from common.otlp.trace.span import Span
from common.service import get_db_service
from common.service.db.db_service import session_getter
from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import ConfigDict
from starlette.responses import StreamingResponse

from agent.api.schemas_v2.bot_chat_inputs import Chat
from agent.api.schemas_v2.bot_dsl import BotDsl
from agent.api.v1.base_api import CompletionBase
from agent.domain.models.bot import BotRelease
from agent.exceptions.agent_exc import AgentInternalExc
from agent.infra.app_auth import APPAuth
from agent.service.builder.chat_builder import ChatRunnerBuilder
from agent.service.runner.debug_chat_runner import DebugChatRunner

from common.otlp.log_trace.node_trace_log import NodeTraceLog as NodeTrace
from common.otlp.metrics.meter import Meter
from common.exceptions.base import BaseExc
from agent.exceptions.agent_exc import AgentInternalExc, AgentNormalExc
from agent.api.base import RunContext
import traceback

chat_router = APIRouter()

headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class CustomChatCompletion(CompletionBase):
    """Custom chat completion for debug chat agents."""

    bot_id: str
    uid: str
    question: str
    dsl: BotDsl
    model_config = ConfigDict(arbitrary_types_allowed=True)
    span: Span

    def __init__(self, inputs: Chat, **data: Any) -> None:
        super().__init__(inputs=inputs, **data)

    async def build_runner(self, span: Span) -> DebugChatRunner:
        """Build ChatRunnerRunner"""
        builder = ChatRunnerBuilder(
            app_id=self.app_id,
            uid=self.uid,
            span=span,
            inputs=cast(Chat, self.inputs),
            dsl=self.dsl,
        )
        return await builder.build()

    async def do_complete(self) -> AsyncGenerator[str, None]:
        """Run agent"""
        with self.span.start("ChatNode") as sp:
            sp.set_attributes(
                attributes={
                    "app_id": self.app_id,
                    "bot_id": self.bot_id,
                    "uid": self.uid,
                }
            )
            sp.add_info_events(
                {"chat-inputs": self.inputs.model_dump_json(by_alias=True)}
            )
            node_trace = await self.build_node_trace(bot_id=self.bot_id, span=sp)
            meter = await self.build_meter(sp)

            # Use parent class run_runner method which includes _finalize_run logic
            async for response in self.run_runner(node_trace, meter, span=sp):
                yield response

    async def run_runner(
            self, node_trace: NodeTrace, meter: Meter, span: Span
    ) -> AsyncGenerator[str, None]:
        with span.start("RunRunner") as sp:
            error: BaseExc = AgentNormalExc()
            error_log: str = ""
            chunk_logs: List[str] = []

            try:
                runner = await self.build_runner(sp)
                if runner is None:
                    raise AgentInternalExc("Failed to build runner")

                async for chunk in runner.run(span=sp, node_trace=node_trace):
                    chunk.id = span.sid
                    async for processed_chunk in self.create_sse_chunk(
                            chunk, chunk_logs
                    ):
                        yield processed_chunk

            except BaseExc as e:
                error = e
                error_log = traceback.format_exc()
            except Exception as e:
                error = AgentInternalExc(str(e))
                error_log = traceback.format_exc()
            finally:
                context = RunContext(
                    error=error,
                    error_log=error_log,
                    chunk_logs=chunk_logs,
                    span=sp,
                    node_trace=node_trace,
                    meter=meter,
                )
                async for final_chunk in self._finalize_run(context):
                    yield final_chunk

    async def create_sse_chunk(
        self, chunk: BotDebugChatCompletionChunk, chunk_logs: List[str]
    ) -> AsyncGenerator[str, None]:
        frame_data = chunk.model_dump_json()
        frame = f"data: {frame_data}\n\n"
        chunk_logs.append(frame_data)
        yield frame


async def _validate_app_auth(app_id: str, sp: Span) -> None:
    app_detail = await APPAuth().app_detail(app_id)
    sp.add_info_events({"kong-app-detail": json.dumps(app_detail, ensure_ascii=False)})
    if app_detail is None:
        raise AgentInternalExc(f"Cannot find appid:{app_id} authentication information")
    if app_detail.get("code") != 0:
        raise AgentInternalExc(app_detail.get("message", ""))
    if len(app_detail.get("data", [])) == 0:
        raise AgentInternalExc(f"Cannot find appid:{app_id} authentication information")


@chat_router.post(  # type: ignore[misc]
    "/bot/chat",
    description="Agent execution - user mode",
    response_model=None,
)
async def bot_chat(
    x_consumer_username: Annotated[str, Header()],
    inputs: Chat,
) -> StreamingResponse:
    """Agent execution - user mode

    Args:
        completion_inputs: Request body
        app_id: Application ID
        bot_id: Bot ID
        uid: User ID
        span: Trace object

    Returns:
        Streaming response
    """

    span = Span(app_id=x_consumer_username)
    with span.start("BotChat") as sp:
        sp.set_attribute("bot_id", inputs.bot_id)
        sp.add_info_events({"bot-chat-inputs": inputs.model_dump_json(by_alias=True)})

        await _validate_app_auth(x_consumer_username, sp)
        with session_getter(get_db_service()) as session:
            # Query Bot table data using bot_id
            bot_release = (
                session.query(BotRelease)
                .filter(
                    BotRelease.bot_id == inputs.bot_id,
                    BotRelease.version == inputs.version_name,
                )
                .first()
            )
            if not bot_release:
                raise AgentInternalExc(
                    f"Bot release with bot_id:{inputs.bot_id} version_name:{inputs.version_name} not found"
                )

            completion = CustomChatCompletion(
                app_id=x_consumer_username,
                inputs=inputs,
                log_caller=inputs.meta_data.caller,
                span=span,
                bot_id="",
                uid=inputs.uid,
                question=inputs.get_last_message_content(),
                dsl=BotDsl(**json.loads(bot_release.dsl)),
            )

            if inputs.stream:
                async def generate() -> AsyncGenerator[str, None]:
                    """Generator for streaming response."""
                    async for response in completion.do_complete():
                        yield response

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers=headers,
                )

            aggregated_content: str = ""
            aggregated_reasoning: str = ""
            aggregated_tool_calls: List[Any] = []
            aggregated_tool_call_responses: List[Any] = []
            usage: Any = None
            code: int = 0
            message: str = "success"
            resp_id: str = ""
            model: str = ""
            created: int = 0

            async for response in completion.do_complete():
                if not response.startswith("data:"):
                    continue
                payload_str = response[len("data:"):].strip()
                if payload_str == "[DONE]":
                    continue
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                if payload.get("object") != "chat.completion.chunk":
                    continue

                code = payload.get("code", code)
                message = payload.get("message", message)
                resp_id = payload.get("id", resp_id)
                model = payload.get("model", model)
                created = payload.get("created", created)

                choices = payload.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                aggregated_content += delta.get("content", "") or ""
                aggregated_reasoning += delta.get("reasoning_content", "") or ""
                aggregated_tool_calls.extend(delta.get("tool_calls") or [])
                aggregated_tool_call_responses.extend(
                    delta.get("tool_call_responses") or []
                )

                if "usage" in payload:
                    usage = payload.get("usage")

            aggregated_payload = {
                "code": code,
                "message": message,
                "id": resp_id,
                "created": created or int(time.time() * 1000),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "delta": {
                            "role": "assistant",
                            "content": aggregated_content,
                            "reasoning_content": aggregated_reasoning,
                            "tool_calls": aggregated_tool_calls,
                            "tool_call_responses": aggregated_tool_call_responses,
                        },
                    }
                ],
                "object": "chat.completion.chunk",
                "model": model,
            }

            if usage is not None:
                aggregated_payload["usage"] = usage

            return JSONResponse(content=aggregated_payload, headers=headers)
