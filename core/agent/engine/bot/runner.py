import asyncio
import json
import time
from pydantic import BaseModel, Field
from common.otlp.trace.span import Span
from common.otlp.log_trace.node_trace_log import NodeTraceLog as NodeTrace
from typing import Any, AsyncIterator, Union
from agent.api.schemas.agent_response import BotAgentResponse
from agent.engine.nodes.base import RunnerBase
from agent.engine.bot.prompt import BOT_TEMPLATE
from agent.service.plugin.base import BasePlugin
from agent.service.plugin.link import LinkPlugin
from agent.service.plugin.mcp import McpPlugin
from agent.service.plugin.workflow import WorkflowPlugin
from agent.service.plugin.knowledge import KnowledgePlugin
from agent.api.schemas.llm_message import LLMMessage, LLMMessages
from common.otlp.log_trace.base import Usage
from common.otlp.log_trace.node_log import Data, NodeLog
from common.exceptions.base import BaseExc
from agent.exceptions.cot_exc import CotFormatIncorrectExc
from common.utils.snowfake import get_id
from agent.exceptions.bot_exc import ReasonStepPluginNotFoundExc


class BotCotStep(BaseModel):
    tool_call_list: list = Field(default_factory=list)
    empty: bool = Field(default=False)


class BotRunner(RunnerBase):
    instruct: str = Field(default="")
    question: str = Field(default="")
    plugins: list[
        Union[BasePlugin, McpPlugin, LinkPlugin, WorkflowPlugin, KnowledgePlugin]
    ]
    max_loop: int = Field(default=12)
    error_tool_call_nums: int = Field(default=4)
    max_read_retry: int = Field(default=3)

    scratchpad_steps: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_number: dict[str, int] = Field(default_factory=dict)

    async def run(
        self, span: Span, node_trace: NodeTrace
    ) -> AsyncIterator[BotAgentResponse]:
        with span.start("RunBotAgent") as sp:
            self.error_tool_call_nums = max(2, self.error_tool_call_nums)

            system_prompt = await self.create_system_prompt()
            user_prompt = await self.create_user_instruct()

            loop_count = 0

            while self.max_loop > loop_count:
                loop_count += 1
                loop_prompt = user_prompt + await self.create_scratchpad_prompt()

                msgs = LLMMessages(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=loop_prompt),
                    ]
                )

                cot_step = BotCotStep(empty=True)
                yield_answer = False
                async for agent_response in self.read_response(
                    msgs, sp, node_trace
                ):
                    if agent_response.typ == "reasoning_content":
                        yield agent_response
                    elif agent_response.typ == "content":
                        yield_answer = True
                        yield agent_response
                    elif agent_response.typ == "cot_step":
                        cot_step = agent_response.content
                        break

                if yield_answer:
                    return
                if cot_step.empty:
                    # todo 降级到Function Call
                    sp.add_info_events({
                        "bot-fallback": "cot_format_incorrect",
                        "reason": "Model returned content format is incorrect",
                    })
                    async for r in self._force_final_answer(sp, node_trace, reason="cot_format_incorrect"):
                        yield r
                    return

                await self.plugin_check(cot_step.tool_call_list)

                # send concurrence msg
                yield BotAgentResponse(typ="tool_call", content=cot_step.tool_call_list)

                try:
                    results = await asyncio.gather(
                        *[
                            self.handle_tool_call(tool_call=tool_call, span=sp)
                            for tool_call in cot_step.tool_call_list
                        ],
                        return_exceptions=True,
                    )

                    exceptions = [r for r in results if isinstance(r, Exception)]
                    if exceptions:
                        for exc in exceptions:
                            sp.record_exception(exc)
                        raise exceptions[0]

                    warn_threshold = self.error_tool_call_nums - 1 if self.error_tool_call_nums > 1 else None
                    results_for_scratchpad = list(results)
                    fallback_first: tuple[str, dict[str, Any], int] | None = None

                    for idx, (tool_call, res) in enumerate(
                        zip(cot_step.tool_call_list, results)
                    ):
                        tool_name = tool_call.get("name", "")
                        args = tool_call.get("arguments", {})
                        if not isinstance(args, dict):
                            raise CotFormatIncorrectExc(
                                m=f"Tool arguments must be an object for {tool_name}"
                            )

                        key = self._tool_call_key(tool_name, args)
                        new_count = self.tool_call_number.get(key, 0) + 1
                        self.tool_call_number[key] = new_count

                        if warn_threshold is not None and new_count == warn_threshold:
                            warn = self._build_repeat_warning(tool_name, args, new_count, res)
                            results_for_scratchpad[idx] = warn

                        fallback_hit = (
                            new_count >= self.error_tool_call_nums if self.error_tool_call_nums > 1 else new_count > self.error_tool_call_nums
                        )
                        if fallback_hit:
                            fallback_obs = self._build_repeat_fallback_observation(
                                tool_name, args, new_count
                            )
                            results_for_scratchpad[idx] = fallback_obs
                            if fallback_first is None:
                                fallback_first = (tool_name, args, new_count)

                    self.scratchpad_steps.append(
                        {
                            "tool_calls": cot_step.tool_call_list,
                            "results": results_for_scratchpad,
                        }
                    )

                    yield BotAgentResponse(typ="tool_call_result", content=results)

                    if fallback_first is not None:
                        sp.add_info_events({
                            "bot-fallback": "tool_call_limit_reached",
                            "tool": fallback_first[0],
                            "count": fallback_first[2],
                        })
                        async for r in self._force_final_answer(sp, node_trace, reason="tool_call_limit_reached"):
                            yield r
                        return

                except BaseExc:
                    raise
                except Exception as e:
                    sp.record_exception(e)
                    raise

            sp.add_info_events({"bot-fallback": "max_loop_exceeded", "max_loop": self.max_loop})
            async for r in self._force_final_answer(sp, node_trace, reason="max_loop_exceeded"):
                yield r
            return

    async def handle_tool_call(self, tool_call: dict, span: Span) -> dict[str, Any]:
        """Execute a single tool call and return a serializable result for Observation."""
        tool_name = tool_call.get("name", "")
        tool_arguments = tool_call.get("arguments", {})
        plugin = tool_call.get("plugin", None)
        tool_id = str(tool_call.get("id", "") or "")

        if not plugin:
            raise ReasonStepPluginNotFoundExc(m=f"Plugin {tool_name} not found")
        if not isinstance(tool_arguments, dict):
            raise CotFormatIncorrectExc(m=f"Tool arguments must be an object for {tool_name}")

        plugin_type = getattr(plugin, "typ", None) or ""

        def ok(payload: Any) -> dict[str, Any]:
            return {
                "id": tool_id,
                "name": tool_name,
                "plugin_type": plugin_type,
                "ok": True,
                "response": payload,
            }

        def fail(err: Exception) -> dict[str, Any]:
            return {
                "id": tool_id,
                "name": tool_name,
                "plugin_type": plugin_type,
                "ok": False,
                "error": str(err),
            }

        with span.start(f"ToolCall:{tool_name}") as sp:
            try:
                supported_types = {"link", "mcp", "workflow", "knowledge"}
                if plugin_type not in supported_types:
                    raise NotImplementedError(f"Unsupported plugin type: {plugin_type}")

                plugin_response = await plugin.run(tool_arguments, sp)
                result = plugin_response.result

                return ok(result)

            except BaseExc:
                raise
            except Exception as e:
                sp.record_exception(e)
                return fail(e)

    async def create_system_prompt(self) -> str:
        prompt = BOT_TEMPLATE.replace("{now}", self.cur_time())
        prompt = prompt.replace(
            "{tools}", "\n".join([tool.schema_template for tool in self.plugins])
        )
        return prompt

    async def create_user_instruct(self) -> str:
        instruct = ""
        if self.instruct:
            instruct += f"用户指令：\n{self.instruct}\n"

        instruct += (
            f"Previous chat history:\n{await self.create_history_prompt()}\n"
            f"Question: {self.question}\n"
        )
        return instruct

    async def create_scratchpad_prompt(self) -> str:
        if not self.scratchpad_steps:
            return ""
        parts: list = []
        for i, step in enumerate(self.scratchpad_steps, start=1):
            tool_calls = step.get("tool_calls", [])
            results = step.get("results", [])
            tool_calls_for_prompt = []
            for tc in tool_calls:
                tool_calls_for_prompt.append(
                    {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                    }
                )

            parts.append(f"Action{i}:")
            parts.append("<|ActionCallBegin|>")
            parts.append(json.dumps(tool_calls_for_prompt, ensure_ascii=False))
            parts.append("<|ActionCallEnd|>")
            parts.append("Observation: " + json.dumps(results, ensure_ascii=False) + "\n")

        return "\n".join(parts) + "\n"

    async def read_response(
            self, messages: LLMMessages, span: Span, node_trace: NodeTrace
    ):
        sp = span

        for attempt in range(self.max_read_retry + 1):
            thinks = ""
            answers = ""

            step_content = ""
            final_answer = False
            tag_found = False

            node_id = ""
            node_sid = span.sid
            node_node_id = span.sid
            node_type = "LLM"
            node_name = "ReadResponse"
            node_start_time = int(round(time.time() * 1000))
            node_running_status = True
            node_data_input = {
                "read_response_input": json.dumps(messages.list(), ensure_ascii=False)
            }
            node_data_output = {}
            node_data_config = {}
            node_data_usage = Usage()

            try:
                async for chunk in self.model.stream(messages.list(), True, sp):
                    delta = chunk.choices[0].delta.dict()
                    reasoning_content = delta.get("reasoning_content", "") or ""
                    content: str = delta.get("content", "") or ""
                    thinks += reasoning_content
                    answers += content

                    node_data_usage.completion_tokens = 0
                    node_data_usage.prompt_tokens = 0
                    node_data_usage.total_tokens = 0
                    if chunk.usage and chunk.usage.model_dump().get("total_tokens") != 0:
                        node_data_usage.completion_tokens = chunk.usage.model_dump().get(
                            "completion_tokens"
                        )
                        node_data_usage.prompt_tokens = chunk.usage.model_dump().get(
                            "prompt_tokens"
                        )
                        node_data_usage.total_tokens = chunk.usage.model_dump().get(
                            "total_tokens"
                        )

                    if reasoning_content:
                        yield BotAgentResponse(
                            typ="reasoning_content",
                            content=reasoning_content,
                            model=self.model.name,
                        )
                        continue

                    if final_answer:
                        if content:
                            yield BotAgentResponse(
                                typ="content",
                                content=content,
                                model=self.model.name,
                            )
                        continue

                    step_content += content
                    if "FinalAnswer:" in step_content:
                        tag_found = True
                        final_answer = True
                        right_content = step_content.split("FinalAnswer:")[1].strip()
                        if right_content:
                            yield BotAgentResponse(
                                typ="content",
                                content=right_content,
                                model=self.model.name,
                            )
                        continue

                    if "<|ActionCallEnd|>" in step_content:
                        tag_found = True
                        bot_cot_step = await self.parse_cot_step(step_content)
                        yield BotAgentResponse(
                            typ="cot_step",
                            content=bot_cot_step,
                            model= self.model.name
                        )
                        break

            except asyncio.CancelledError:
                node_running_status = False
                raise
            except Exception as e:
                if attempt < self.max_read_retry:
                    sp.add_info_events({
                        "read_response_retry": attempt + 1,
                        "reason": "exception",
                        "error": str(e)[:200],
                    })
                    continue
                raise
            finally:
                node_end_time = int(round(time.time() * 1000))
                node_trace.trace.append(
                    NodeLog(
                        id=node_id,
                        sid=node_sid,
                        node_id=node_node_id,
                        node_name=node_name,
                        node_type=node_type,
                        start_time=node_start_time,
                        end_time=node_end_time,
                        duration=node_end_time - node_start_time,
                        running_status=node_running_status,
                        data=Data(
                            input=node_data_input if node_data_input else {},
                            output=node_data_output if node_data_output else {},
                            config=node_data_config if node_data_config else {},
                            usage=node_data_usage,
                        ),
                    )
                )

                sp.add_info_events({"step-think": thinks})
                sp.add_info_events({"step-content": answers})

            if tag_found:
                return

            if attempt < self.max_read_retry:
                sp.add_info_events({
                    "read_response_retry": attempt + 1,
                    "reason": "no_final_or_cot",
                })
                continue

            if not tag_found:
                yield BotAgentResponse(
                    typ="cot_step",
                    content=BotCotStep(empty=True),
                    model = self.model.name
                )
                return

    async def parse_cot_step(self, step_content: str) -> BotCotStep:
        if not (
            "<|ActionCallBegin|>" in step_content
            and "<|ActionCallEnd|>" in step_content
        ):
            raise CotFormatIncorrectExc(m="Model returned content format is incorrect")
        action_content = step_content.split("<|ActionCallBegin|>")[1].split(
            "<|ActionCallEnd|>"
        )[0]
        try:
            tool_call_list = json.loads(action_content)
        except Exception:
            raise CotFormatIncorrectExc(m="Model returned content format is incorrect")

        if not isinstance(tool_call_list, list) or not all(
            isinstance(i, dict) for i in tool_call_list
        ):
            raise CotFormatIncorrectExc(m="Model returned content format is incorrect")
        for tool_call in tool_call_list:
            if "name" not in tool_call:
                raise CotFormatIncorrectExc(m="Model returned content format is incorrect")
            if "arguments" not in tool_call:
                tool_call["arguments"] = {}
            tool_call["id"] = get_id()

        return BotCotStep(tool_call_list=tool_call_list)

    async def get_plugin(self, tool_name: str):
        for plugin in self.plugins:
            if plugin.name == tool_name:
                return plugin
        return None

    async def plugin_check(self, tool_call_list: list):
        for tool_call in tool_call_list:
            tool_name = tool_call.get("name", "")
            plugin = await self.get_plugin(tool_name)
            if not plugin:
                raise ReasonStepPluginNotFoundExc(m=f"Plugin {tool_name} not found")
            tool_call["plugin"] = plugin

    def _normalize_tool_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, dict):
            return {k: self._normalize_tool_arguments(arguments[k]) for k in sorted(arguments.keys())}
        if isinstance(arguments, list):
            return [self._normalize_tool_arguments(i) for i in arguments]
        if isinstance(arguments, tuple):
            return [self._normalize_tool_arguments(i) for i in arguments]
        if isinstance(arguments, (str, int, float, bool)) or arguments is None:
            return arguments
        return str(arguments)

    def _tool_call_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        normalized = self._normalize_tool_arguments(arguments)
        try:
            payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            payload = str(normalized)
        return f"{tool_name}:{payload}"

    def _build_repeat_warning(self, tool_name: str, arguments: dict[str, Any], count: int, result: Any | None = None) -> dict[str, Any]:
        return {
            "ok": True,
            "warning": True,
            "type": "repeat_tool_call_warning",
            "tool": tool_name,
            "arguments": arguments,
            "count": count,
            "error_tool_call_nums": self.error_tool_call_nums,
            "result": result,
            "warn_message": (
                f"检测到同一个工具使用相同参数被重复调用：{tool_name}。"
                f"当前重复次数={count}，最大允许次数={count}，已达到使用上限。"
                "请避免用相同参数反复调用同一个工具；优先复用已有 Observation。"
            ),
        }

    def _build_repeat_fallback_observation(self, tool_name: str, arguments: dict[str, Any], count: int) -> dict[str, Any]:
        return {
            "ok": False,
            "fallback": True,
            "type": "repeat_tool_call_limit_reached",
            "tool": tool_name,
            "arguments": arguments,
            "count": count,
            "error_tool_call_nums": self.error_tool_call_nums,
            "message": (
                f"同一个工具使用相同参数的重复调用次数已超过上限：{tool_name}。"
                "系统将停止继续调用工具。请基于已有 Observation 直接给出 FinalAnswer:"
            ),
        }

    async def _force_final_answer(self, span: Span, node_trace: NodeTrace, reason: str = "") -> AsyncIterator[BotAgentResponse]:
        with span.start("ForceFinalAnswer") as sp:
            sp.add_info_events({"force-final-reason": reason or "unknown"})
            reason_text = {
                "tool_call_limit_reached": "工具重复调用达到上限。",
                "max_loop_exceeded": f"循环次数已达到上限（max_loop={self.max_loop}）。",
                "cot_format_incorrect": "模型返回内容格式不正确",
            }.get(reason, "触发兜底回复。")

            force_system_prompt = (
                f"{reason_text}工具调用已被禁用。你必须基于当前已有信息进行回答。"
                "输出必须以 'FinalAnswer:' 开头，并且禁止输出任何 ActionCall 相关标签。"
            )

            system_prompt = await self.create_system_prompt() + force_system_prompt
            user_prompt = await self.create_user_instruct() + await self.create_scratchpad_prompt()

            msgs = LLMMessages(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ]
            )

            async for agent_response in self.read_response(msgs, sp, node_trace):
                if agent_response.typ in {"reasoning_content", "content"}:
                    yield agent_response
                    continue
                if agent_response.typ == "cot_step":
                    sp.add_info_events({
                        "force-final-error": "cot_step_returned",
                        "message": "Model attempted tool calls after tool invocation was disabled.",
                    })
                    raise CotFormatIncorrectExc(
                        m="Model attempted tool calls after tool invocation was disabled"
                    )

