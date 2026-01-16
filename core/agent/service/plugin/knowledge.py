import time
import asyncio
import json
import os
from typing import Any, Dict, List

import aiohttp
from common.otlp.trace.span import Span
from openai import BaseModel

from agent.exceptions.plugin_exc import KnowledgeQueryExc, PluginExc
from agent.service.plugin.base import BasePlugin, PluginResponse


async def chunk_query(
    query: str,
    top_k: int,
    repo_ids: List[str],
    doc_ids: List[str],
    score_threshold: float,
    rag_type: str,
    span: Span,
) -> Dict[str, Any]:
    with span.start("ChunkQuery") as sp:
        data: Dict[str, Any] = {
            "query": query,
            "topN": str(top_k),
            "match": {"repoId": repo_ids, "threshold": score_threshold},
            "ragType": rag_type,
        }
        if rag_type == "CBG-RAG":
            if "match" not in data:
                data["match"] = {}
            data["match"]["docIds"] = doc_ids

        sp.add_info_events({"request-data": json.dumps(data, ensure_ascii=False)})

        if not repo_ids:
            empty_resp: Dict[str, Any] = {}
            sp.add_info_events(
                {"response-data": json.dumps(empty_resp, ensure_ascii=False)}
            )
            return empty_resp

        try:
            query_url = os.getenv("CHUNK_QUERY_URL")
            if not query_url:
                raise KnowledgeQueryExc("CHUNK_QUERY_URL is not set")
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=40)
                async with session.post(
                    query_url, json=data, timeout=timeout
                ) as response:

                    sp.add_info_events({"response-data": str(await response.read())})

                    response.raise_for_status()
                    if response.status == 200:
                        resp: Dict[str, Any] = await response.json()
                        sp.add_info_events(
                            {"response-data": json.dumps(resp, ensure_ascii=False)}
                        )
                        return resp

                    raise KnowledgeQueryExc
        except asyncio.TimeoutError as e:
            raise KnowledgeQueryExc from e


def process_chunk_query_result(
    resp: Dict[str, Any], repo_type: str
) -> List[Dict[str, Any]]:
    metadata_list = []
    for result in resp.get("data", {}).get("results", []):
        score = result.get("score", 0)
        content = result.get("content", "")
        references = result.get("references", {})
        for ref_key, ref_value in references.items():
            if repo_type == "AIUI-RAG2":
                ref_format = ref_value.get("format", "")
                if ref_format == "image":
                    content = content.replace(
                        f"<{ref_key}>", f"![alt]({ref_value.get('link', '')})"
                    )
                elif ref_format == "table":
                    content = content.replace(
                        f"<{ref_key}>", f"\n{ref_value.get('content', '')}\n"
                    )
            if repo_type == "CBG-RAG":
                content = content.replace(f"{{{ref_key}}}", f"![alt]({ref_value})")
        metadata_list.append({"score": score, "chunk_content": content})
    return metadata_list


class KnowledgePluginRunner(BaseModel):
    top_k: int
    repo_ids: List[str]
    doc_ids: List[str]
    score_threshold: float
    rag_type: str

    async def run(self, action_input: dict[str, str], span: Span) -> PluginResponse:
        start_time = int(round(time.time() * 1000))
        resp = await chunk_query(
            action_input.get("query", ""),
            self.top_k,
            self.repo_ids,
            self.doc_ids,
            self.score_threshold,
            self.rag_type,
            span,
        )
        metadata_list = process_chunk_query_result(resp, self.rag_type)
        end_time = int(round(time.time() * 1000))
        return PluginResponse(
            code=resp.get("code", 0),
            sid=span.sid,
            start_time=start_time,
            end_time=end_time,
            result=metadata_list,
        )


class KnowledgePlugin(BasePlugin):
    pass


class KnowledgePluginFactory(BaseModel):
    query: str
    top_k: int
    repo_ids: List[str]
    doc_ids: List[str]
    score_threshold: float
    rag_type: str

    def gen(self) -> KnowledgePlugin:
        return KnowledgePlugin(
            name="knowledge",
            description="knowledge plugin",
            schema_template="",
            typ="knowledge",
            run=self.retrieve,
        )

    async def retrieve(self, span: Span) -> Dict[str, Any]:
        with span.start("retrieve") as sp:

            return await chunk_query(
                self.query,
                self.top_k,
                self.repo_ids,
                self.doc_ids,
                self.score_threshold,
                self.rag_type,
                span,
            )
