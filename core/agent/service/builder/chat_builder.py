import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

from agent.engine.bot.runner import BotRunner
from common.otlp.trace.span import Span

from agent.api.schemas_v2.bot_chat_inputs import Chat
from agent.api.schemas_v2.bot_dsl import BotDsl, RagKnowledge
from agent.service.builder.base_builder import (
    BaseApiBuilder,
    CotRunnerParams,
    RunnerParams,
)
from agent.service.plugin.knowledge import KnowledgePluginFactory
from agent.service.runner.debug_chat_runner import DebugChatRunner
from agent.service.plugin.bot_plugins import BotPluginFactory


@dataclass
class KnowledgeQueryParams:
    """Knowledge query parameters"""

    repo_ids: list[str]
    doc_ids: list[str]
    top_k: int
    score_threshold: float
    rag_type: str


class ChatRunnerBuilder(BaseApiBuilder):
    inputs: Chat
    dsl: BotDsl

    async def build(self) -> DebugChatRunner:
        """Build"""
        with self.span.start("BuildRunner") as sp:
            model = await self.create_model(
                app_id=self.app_id,
                model_name=self.dsl.model.properties.id,
                base_url=self.dsl.model.properties.url,
                api_key=self.dsl.model.properties.bearer_token,
            )

            plugins = await BotPluginFactory(
                app_id=self.app_id,
                uid=self.uid,
                dsl=self.dsl).gen_tools(sp)

            return DebugChatRunner(
                bot_runner=BotRunner(
                    model=model,
                    chat_history=self.inputs.get_chat_history(),
                    plugins=plugins,
                    instruct=self.dsl.prompt.instruct,
                    question=self.inputs.get_last_message_content()
                )
            )

    async def query_knowledge_by_workflow(
        self, knowledge_list: list[RagKnowledge], span: Span
    ) -> tuple[list, str]:
        """Query knowledge base"""
        with span.start("QueryKnowledgeByWorkflow") as sp:
            tasks = self._create_knowledge_tasks(knowledge_list, sp)

            if not tasks:
                return [], ""

            results = await asyncio.gather(*tasks)
            metadata_list, _ = self._process_knowledge_results(results)
            backgrounds = self._extract_backgrounds(metadata_list)

            sp.add_info_events(
                {
                    "metadata-list": json.dumps(metadata_list, ensure_ascii=False),
                    "backgrounds": backgrounds,
                }
            )

            return metadata_list, backgrounds

    def _create_knowledge_tasks(
        self, knowledge_list: list[RagKnowledge], span: Span
    ) -> list:
        """Create knowledge query tasks"""
        tasks = []

        for knowledge in knowledge_list:
            repo_ids = knowledge.properties.repos or []
            doc_ids = knowledge.properties.docs or []

            # Add debug logs
            span.add_info_events(
                {
                    "knowledge_name": knowledge.name,
                    "repo_type": knowledge.repo_type,
                    "repo_ids": repo_ids,
                    "doc_ids": doc_ids,
                }
            )

            if not (repo_ids or doc_ids):
                span.add_info_events({"skip_reason": "no repo_ids or doc_ids"})
                continue

            top_k = knowledge.properties.top_k or 3
            score_threshold = knowledge.properties.min_score or 0.3
            repo_type = knowledge.type

            # Add mapped logs
            span.add_info_events({"mapped_rag_type": repo_type})

            params = KnowledgeQueryParams(
                repo_ids=repo_ids,
                doc_ids=doc_ids,
                top_k=top_k,
                score_threshold=score_threshold,
                rag_type=repo_type,
            )
            task = self.exec_query_knowledge(params, span)
            tasks.append(task)

        return tasks

    def _process_knowledge_results(
        self, results: list
    ) -> tuple[list, dict[str, list[dict[str, str]]]]:
        """Process knowledge query results"""
        metadata_list = []
        metadata_map: dict[str, list[dict[str, str]]] = {}

        for resp in results:
            for result in resp.get("data", {}).get("results", []):
                title = result.get("title", "")
                source_id = result.get("docId", "")
                content = result.get("content", "")
                references = result.get("references", {})

                content = self._process_content_references(content, references)

                if source_id not in metadata_map:
                    metadata_map[source_id] = []
                metadata_map[source_id].append({"chunk_context": f"{title}\n{content}"})

        for source_id, metadata in metadata_map.items():
            metadata_list.append({"source_id": source_id, "chunk": metadata})

        return metadata_list, metadata_map

    def _process_content_references(self, content: str, references: dict) -> str:
        """Process references in content"""
        for ref_key, ref_value in references.items():
            if isinstance(ref_value, dict):
                ref_format = ref_value.get("format", "")
                if ref_format == "image":
                    content = content.replace(
                        f"<{ref_key}>",
                        f"![alt]({ref_value.get('link', '')})",
                    )
                elif ref_format == "table":
                    content = content.replace(
                        f"<{ref_key}>",
                        f"\n{ref_value.get('content', '')}\n",
                    )
            else:
                content = content.replace(f"{{{ref_key}}}", f"![alt]({ref_value})")
        return content

    def _extract_backgrounds(self, metadata_list: list) -> str:
        """Extract background information"""
        background_list = []
        for metadata in metadata_list:
            chunk = metadata.get("chunk", [])
            for c in chunk:
                bg = c.get("chunk_context", "")
                background_list.append(bg)
        return "\n".join(background_list)

    async def exec_query_knowledge(
        self,
        params: KnowledgeQueryParams,
        span: Span,
    ) -> dict[str, Any]:
        knowledge_plugin = KnowledgePluginFactory(
            query=self.inputs.get_last_message_content(),
            top_k=params.top_k,
            repo_ids=params.repo_ids,
            doc_ids=params.doc_ids,
            score_threshold=params.score_threshold,
            rag_type=params.rag_type,
        ).gen()

        resp = await knowledge_plugin.run(span=span)

        return cast(dict[str, Any], resp)
