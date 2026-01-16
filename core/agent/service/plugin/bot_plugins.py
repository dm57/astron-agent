import aiohttp
import os
import copy
from pydantic import BaseModel, Field
from typing import Union, Any
import json
from agent.service.plugin.base import BasePlugin, PluginResponse
from agent.service.plugin.mcp import McpPlugin, McpPluginRunner
from agent.service.plugin.workflow import WorkflowPlugin, WorkflowPluginRunner
from agent.service.plugin.knowledge import KnowledgePlugin, KnowledgePluginRunner
from agent.service.plugin.link import LinkPlugin, LinkPluginRunner
from agent.api.schemas_v2.bot_dsl import BotDsl, LinkMcpServerInputs, CusMcpServerInputs
from common.otlp.trace.span import Span
from common.service import get_db_service
from common.service.db.db_service import session_getter
from agent.domain.models.bot import ToolOperationDetail
from agent.exceptions.plugin_exc import GetToolSchemaExc


class BotLinkRunner(LinkPluginRunner):

    map_keys: dict[str, str] = Field(default_factory=dict)

    async def run(self, action_input: dict[str, Any], span: Span) -> PluginResponse:
        tool_input = copy.deepcopy(action_input)
        for k, v in self.map_keys.items():
            if v in tool_input and k != v:
                tool_input[k] = tool_input[v]
                del tool_input[v]
        result = await super().run(tool_input, span)
        return result


class BotMcpRunner(McpPluginRunner):
    map_keys: dict[str, str] = Field(default_factory=dict)

    async def run(self, action_input: dict[str, Any], span: Span) -> PluginResponse:
        tool_input = copy.deepcopy(action_input)
        for k, v in self.map_keys.items():
            if v in tool_input and k != v:
                tool_input[k] = tool_input[v]
                del tool_input[v]
        result = await super().run(tool_input, span)
        return result


class BotWorkflowRunner(WorkflowPluginRunner):
    map_keys: dict[str, str] = Field(default_factory=dict)

    async def run(self, action_input: dict[str, Any], span: Span) -> PluginResponse:
        tool_input = copy.deepcopy(action_input)
        for k, v in self.map_keys.items():
            if v in tool_input and k != v:
                tool_input[k] = tool_input[v]
                del tool_input[v]

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        last_resp: PluginResponse | None = None

        async for resp in super().run(tool_input, span):
            last_resp = resp
            if isinstance(resp.result, dict):
                c = resp.result.get("content")
                r = resp.result.get("reasoning_content")
                if isinstance(c, str) and c:
                    content_parts.append(c)
                if isinstance(r, str) and r:
                    reasoning_parts.append(r)

        if last_resp is None:
            return PluginResponse(
                code=0,
                sid="",
                start_time=0,
                end_time=0,
                result={"content": "", "reasoning_content": ""},
                log=[],
            )

        if content_parts or reasoning_parts:
            return last_resp.model_copy(
                update={
                    "result": {
                        "content": "".join(content_parts),
                        "reasoning_content": "".join(reasoning_parts),
                    }
                }
            )

        return last_resp


class BotPluginFactory(BaseModel):
    app_id: str = Field(...)
    uid: str = Field(...)
    dsl: BotDsl = Field(...)

    async def gen_tools(
        self, span: Span
    ) -> list[
        Union[BasePlugin, McpPlugin, LinkPlugin, WorkflowPlugin, KnowledgePlugin]
    ]:
        with span.start("GenTools") as sp:
            tools = []
            tools.extend(await self.gen_link_tools(sp))
            tools.extend(await self.gen_mcp_tools(sp))
            tools.extend(await self.gen_workflow_tools(sp))
            tools.extend(await self.gen_knowledge_tools(sp))
            return tools

    async def gen_link_tools(self, span: Span) -> list[LinkPlugin]:

        with span.start("GenLinkTools") as sp:
            tools = []
            for tool in self.dsl.plugin.link_tools:
                link_plugin = LinkPlugin(
                    tool_id=tool.id,
                    name=tool.fc_schema.name,
                    description=tool.fc_schema.description,
                    schema_template=json.dumps(
                        {
                            "name": tool.fc_schema.name,
                            "description": tool.fc_schema.description,
                            "parameters": {
                                "type": tool.fc_schema.parameters.type,
                                "properties": {  # filter name field
                                    v.name: {
                                        "description": v.description,
                                        "type": v.type,
                                    }
                                    for _, v in tool.fc_schema.parameters.properties.items()
                                },
                                "required": [
                                    v.name
                                    for r in tool.fc_schema.parameters.required
                                    for k, v in tool.fc_schema.parameters.properties.items()
                                    if k == r
                                ],
                            },
                        },
                        ensure_ascii=False
                    ),
                    typ="link",
                    run=BotLinkRunner(
                        app_id=self.app_id,
                        uid=self.uid,
                        tool_id=tool.id,
                        version=tool.version,
                        operation_id=tool.operation_id,
                        method_schema=await self.get_link_method_schema_cache(
                            tool.id, tool.version, tool.operation_id, sp
                        ),
                        map_keys={
                            k: v.name
                            for k, v in tool.fc_schema.parameters.properties.items()
                        },
                    ).run,
                )
                tools.append(link_plugin)
            return tools

    async def gen_mcp_tools(self, span: Span) -> list[McpPlugin]:
        with span.start("GenMcpTools") as sp:
            tools = []
            McpServer = Union[LinkMcpServerInputs, CusMcpServerInputs]
            servers: list[McpServer] = [
                *self.dsl.plugin.link_mcp_servers,
                *self.dsl.plugin.cus_mcp_servers
            ]
            for server in servers:
                for tool in server.tools:
                    server_id = ""
                    server_url = ""
                    if isinstance(server, LinkMcpServerInputs):
                        server_id = server.server_id
                    elif isinstance(server, CusMcpServerInputs):
                        server_url = server.server_url
                    mcp_plugin = McpPlugin(
                        server_id=server_id,
                        server_url=server_url,
                        name=tool.fc_schema.name,
                        description=tool.fc_schema.description,
                        schema_template=json.dumps(
                            {
                                "name": tool.fc_schema.name,
                                "description": tool.fc_schema.description,
                                "parameters": {
                                    "type": tool.fc_schema.parameters.type,
                                    "properties": {
                                        v.name: {
                                            "description": v.description,
                                            "type": v.type,
                                        }
                                        for _, v in tool.fc_schema.parameters.properties.items()
                                    },
                                    "required": [
                                        v.name
                                        for r in tool.fc_schema.parameters.required
                                        for k, v in tool.fc_schema.parameters.properties.items()
                                        if k == r
                                    ],
                                },
                            },
                            ensure_ascii=False
                        ),
                        typ="mcp",
                        run=BotMcpRunner(
                            server_id=server_id,
                            server_url=server_url,
                            sid=span.sid,
                            name=tool.operation_id,
                            map_keys={
                                k: v.name
                                for k, v in tool.fc_schema.parameters.properties.items()
                            },
                        ).run,
                    )
                    tools.append(mcp_plugin)
            return tools

    async def gen_workflow_tools(self, span: Span) -> list[WorkflowPlugin]:
        with span.start("GenWorkflowTools") as sp:
            tools = []
            for workflow in self.dsl.plugin.workflows:
                workflow_plugin = WorkflowPlugin(
                    flow_id=workflow.flow_id,
                    name=workflow.fc_schema.name,
                    description=workflow.fc_schema.description,
                    schema_template=json.dumps(
                        {
                            "name": workflow.fc_schema.name,
                            "description": workflow.fc_schema.description,
                            "parameters": {
                                "type": workflow.fc_schema.parameters.type,
                                "properties": {
                                    v.name: {
                                        "description": v.description,
                                        "type": v.type,
                                    }
                                    for _, v in workflow.fc_schema.parameters.properties.items()
                                },
                                "required": [
                                    v.name
                                    for r in workflow.fc_schema.parameters.required
                                    for k, v in workflow.fc_schema.parameters.properties.items()
                                    if k == r
                                ],
                            },
                        },
                        ensure_ascii=False
                    ),
                    typ="workflow",
                    run=BotWorkflowRunner(
                        app_id=self.app_id,
                        uid=self.uid,
                        flow_id=workflow.flow_id,
                        map_keys={
                            k: v.name
                            for k, v in workflow.fc_schema.parameters.properties.items()
                        }
                    ).run,
                )
                tools.append(workflow_plugin)
            return tools

    async def gen_knowledge_tools(self, span: Span) -> list[KnowledgePlugin]:

        with span.start("GenKnowledgeTools") as sp:
            tools = []
            for knowledge in self.dsl.rag.knowledges:
                knowledge_plugin = KnowledgePlugin(
                    name=knowledge.name,
                    description=knowledge.description,
                    schema_template=json.dumps(
                        {
                            "name": knowledge.name,
                            "description": knowledge.description,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The query to search the knowledge base",
                                    },
                                },
                                "required": ["query"],
                            },
                        }
                    ),
                    typ="knowledge",
                    run=KnowledgePluginRunner(
                        top_k=knowledge.properties.top_k,
                        repo_ids=knowledge.properties.repos,
                        doc_ids=knowledge.properties.docs,
                        score_threshold=knowledge.properties.min_score,
                        rag_type=knowledge.type,
                    ).run,
                )
                tools.append(knowledge_plugin)
            return tools

    async def get_link_method_schema_cache(
        self, tool_id: str, version: str, operation_id: str, span: Span
    ) -> dict:
        with span.start("GetLinkMethodSchemaCache") as sp:
            with session_getter(get_db_service()) as session:
                tool_operation_detail = (
                    session.query(ToolOperationDetail)
                    .filter(
                        ToolOperationDetail.tool_id == tool_id,
                        ToolOperationDetail.operation_id == operation_id,
                        ToolOperationDetail.version == version,
                    )
                    .first()
                )
                if tool_operation_detail:
                    return json.loads(tool_operation_detail.method_schema)

                remote_tool_data = await self.get_link_method_schema_remote(
                    tool_id, version, operation_id, sp
                )
                tool_schema_data = json.loads(remote_tool_data.get("schema", "{}"))
                for _, path_schema in tool_schema_data.get("paths", {}).items():
                    for _, method_schema in path_schema.items():
                        if method_schema.get("operationId") == operation_id:
                            method_schema_str = json.dumps(
                                method_schema, ensure_ascii=False
                            )
                            tool_operation_detail = ToolOperationDetail(
                                tool_id=tool_id,
                                operation_id=operation_id,
                                version=version,
                                method_schema=method_schema_str,
                            )
                            session.add(tool_operation_detail)
                            session.commit()
                            return method_schema
                raise GetToolSchemaExc(
                    f"Link: tool not found, tool_id: {tool_id}, operation_id: {operation_id}, version: {version}"
                )

    async def get_link_method_schema_remote(
        self, tool_id: str, version: str, operation_id: str, span: Span
    ) -> dict:
        with span.start("GetLinkMethodSchemaRemote") as sp:
            url = (
                os.getenv("VERSIONS_LINK_URL", "")
                + "?"
                + f"app_id={self.app_id}&tool_ids={tool_id}&versions={version}"
            )
            sp.add_info_events(attributes={"api": url})
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    if response.status == 200:
                        result = await response.json()
                        sp.add_info_events(
                            attributes={
                                "api-outputs": (json.dumps(result, ensure_ascii=False))
                            }
                        )
                        print(result)
                        if result.get("code") != 0:
                            raise GetToolSchemaExc("Link:" + result.get("message", ""))

                        tools_data = result.get("data", {}).get("tools", [])
                        if len(tools_data) == 0:
                            raise GetToolSchemaExc("Link: tools data is empty")
                        return tools_data[0]

                    sp.add_info_events(
                        attributes={
                            "api-outputs": (f"response code is {response.status}")
                        }
                    )
                    raise GetToolSchemaExc(f"Link: response code is {response.status}")
