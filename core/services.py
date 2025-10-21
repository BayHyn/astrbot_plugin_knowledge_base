# astrbot_plugin_knowledge_base/core/services.py
"""服务层 - 封装核心业务逻辑"""
from typing import List, Tuple, Optional, TYPE_CHECKING
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from .constants import (
    KB_START_MARKER,
    KB_END_MARKER,
    USER_PROMPT_DELIMITER_IN_HISTORY,
    QUERY_LOG_LENGTH,
    CONTENT_PREVIEW_LENGTH,
    PROMPT_PREVIEW_LENGTH,
)
from .domain import CollectionMetadataRepository
from ..utils.logging_helper import log_start, log_success, log_error, log_warning, log_debug

if TYPE_CHECKING:
    from ..vector_store.base import VectorDBBase, Document
    from .config_manager import ConfigManager


class SearchService:
    """知识库搜索服务"""

    def __init__(
        self,
        vector_db: "VectorDBBase",
        config_manager: "ConfigManager",
    ):
        """
        初始化搜索服务

        Args:
            vector_db: 向量数据库实例
            config_manager: 配置管理器
        """
        self.vector_db = vector_db
        self.config_manager = config_manager

    async def search(
        self,
        collection_name: str,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple["Document", float]]:
        """
        在知识库中搜索相关文档

        Args:
            collection_name: 知识库名称
            query: 查询字符串
            top_k: 返回结果数量（None则使用配置）
            min_similarity: 最小相似度阈值（None则使用配置）

        Returns:
            List[Tuple[Document, float]]: 文档和相似度分数的列表

        Raises:
            ValueError: 知识库不存在
        """
        # 验证知识库是否存在
        if not await self.vector_db.collection_exists(collection_name):
            raise ValueError(f"知识库 '{collection_name}' 不存在")

        # 使用配置的默认值
        kb_config = self.config_manager.kb_config
        if top_k is None:
            top_k = kb_config.search_top_k
        if min_similarity is None:
            min_similarity = kb_config.min_similarity_score

        log_debug(
            "搜索参数",
            {
                "collection": collection_name,
                "query": query[:QUERY_LOG_LENGTH],
                "top_k": top_k,
                "min_similarity": min_similarity,
            },
        )

        # 执行搜索
        try:
            results = await self.vector_db.search(collection_name, query, top_k=top_k)
            log_success(
                "向量搜索",
                collection_name,
                f"检索到 {len(results)} 个文档",
            )
        except Exception as e:
            log_error("向量搜索", e, collection_name)
            raise

        # 过滤低相似度结果
        filtered_results = [
            (doc, score) for doc, score in results if score >= min_similarity
        ]

        if len(filtered_results) < len(results):
            log_debug(
                "相似度过滤",
                {
                    "保留": len(filtered_results),
                    "总数": len(results),
                    "阈值": min_similarity,
                },
            )

        return filtered_results


class RAGService:
    """RAG (检索增强生成) 服务"""

    def __init__(
        self,
        search_service: SearchService,
        metadata_repo: CollectionMetadataRepository,
        config_manager: "ConfigManager",
    ):
        """
        初始化 RAG 服务

        Args:
            search_service: 搜索服务实例
            metadata_repo: 集合元数据仓库
            config_manager: 配置管理器
        """
        self.search_service = search_service
        self.metadata_repo = metadata_repo
        self.config_manager = config_manager

    def clean_kb_content_from_contexts(self, req: ProviderRequest) -> None:
        """
        清理历史对话中的知识库内容

        Args:
            req: LLM 请求对象,会直接修改其 contexts 属性
        """
        if not req.contexts:
            return

        cleaned_contexts = []
        initial_count = len(req.contexts)

        for message in req.contexts:
            role = message.get("role")
            content = message.get("content", "")

            # 处理 system 消息
            if role == "system" and KB_START_MARKER in content:
                log_debug(
                    "清理历史对话",
                    {"action": "删除知识库 system 消息", "content_preview": content[:CONTENT_PREVIEW_LENGTH]}
                )
                continue

            # 处理 user 消息
            elif role == "user" and KB_START_MARKER in content:
                start_idx = content.find(KB_START_MARKER)
                end_idx = content.find(KB_END_MARKER, start_idx)

                if start_idx != -1 and end_idx != -1:
                    delimiter_idx = content.find(
                        USER_PROMPT_DELIMITER_IN_HISTORY,
                        end_idx + len(KB_END_MARKER),
                    )

                    if delimiter_idx != -1:
                        # 提取原始用户问题
                        original_prompt = content[
                            delimiter_idx + len(USER_PROMPT_DELIMITER_IN_HISTORY) :
                        ].strip()
                        message["content"] = original_prompt
                        cleaned_contexts.append(message)
                        log_debug(
                            "清理历史对话",
                            {"action": "从 user 消息中清理知识库内容", "原问题": original_prompt[:CONTENT_PREVIEW_LENGTH]}
                        )
                    else:
                        log_warning(
                            "清理历史对话",
                            "缺少原始问题分隔符",
                            details={"content_preview": content[:CONTENT_PREVIEW_LENGTH]}
                        )
                        continue
                else:
                    log_warning(
                        "清理历史对话",
                        "缺少知识库结束标记",
                        details={"content_preview": content[:CONTENT_PREVIEW_LENGTH]}
                    )
                    continue
            else:
                # 其他消息保持不变
                cleaned_contexts.append(message)

        req.contexts = cleaned_contexts
        removed_count = initial_count - len(req.contexts)

        if removed_count > 0:
            log_success(
                "清理历史对话",
                result=f"删除了 {removed_count} 条知识库补充消息",
                details={"初始数量": initial_count, "清理后": len(req.contexts)}
            )

    async def enhance_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        collection_name: str,
    ) -> None:
        """
        使用知识库内容增强 LLM 请求

        Args:
            event: 消息事件
            req: LLM 请求对象,会直接修改其内容
            collection_name: 知识库名称

        Raises:
            ValueError: 知识库不存在
        """
        # 获取用户配置
        user_config = self.config_manager.get_user_kb_config(event, collection_name)

        user_query = req.prompt
        if not user_query or not user_query.strip():
            log_debug("知识库增强", {"状态": "跳过", "原因": "用户查询为空"})
            return

        # 搜索相关知识
        try:
            log_start(
                "知识库增强搜索",
                collection_name,
                {"查询": user_query[:QUERY_LOG_LENGTH], "top_k": user_config.search_top_k}
            )
            search_results = await self.search_service.search(
                collection_name=collection_name,
                query=user_query,
                top_k=user_config.search_top_k,
                min_similarity=user_config.min_similarity_score,
            )
        except Exception as e:
            log_error("知识库增强搜索", e, collection_name)
            return

        if not search_results:
            log_warning(
                "知识库增强搜索",
                "未找到相关内容",
                collection_name,
                {"查询": user_query[:QUERY_LOG_LENGTH]}
            )
            return

        log_success(
            "知识库增强搜索",
            collection_name,
            f"检索到 {len(search_results)} 个相关文档",
            {"准备": "注入到 LLM 请求"}
        )

        # 格式化检索内容
        retrieved_contexts_list = []
        for idx, (doc, score) in enumerate(search_results):
            source_info = doc.metadata.get("source", "未知来源")
            context_item = (
                f"- 内容: {doc.text_content} (来源: {source_info}, 相关度: {score:.2f})"
            )
            retrieved_contexts_list.append(context_item)
            log_debug(
                "检索文档详情",
                {
                    "序号": idx+1,
                    "相关度": f"{score:.4f}",
                    "来源": source_info,
                    "内容": doc.text_content[:QUERY_LOG_LENGTH]
                }
            )

        formatted_contexts = "\n".join(retrieved_contexts_list)
        knowledge_to_insert = user_config.context_template.format(
            retrieved_contexts=formatted_contexts
        )

        # 检查长度限制
        max_length = user_config.max_insert_length
        original_length = len(knowledge_to_insert)

        if original_length > max_length:
            log_warning(
                "知识库内容长度检查",
                f"内容过长,截断至 {max_length} chars",
                details={"原长度": original_length, "限制": max_length}
            )
            knowledge_to_insert = (
                knowledge_to_insert[:max_length] + "\n... [内容已截断]"
            )
        else:
            log_debug(
                "知识库内容长度检查",
                {"长度": original_length, "限制": max_length, "状态": "通过"}
            )

        # 添加标记
        knowledge_to_insert = (
            f"{KB_START_MARKER}\n{knowledge_to_insert}\n{KB_END_MARKER}"
        )

        # 根据配置插入内容
        insertion_method = user_config.insertion_method

        if insertion_method == "system_prompt":
            if req.system_prompt:
                req.system_prompt = f"{knowledge_to_insert}\n\n{req.system_prompt}"
            else:
                req.system_prompt = knowledge_to_insert
            log_success(
                "知识库内容注入",
                "system_prompt",
                details={
                    "最终长度": len(req.system_prompt),
                    "知识库部分": len(knowledge_to_insert)
                }
            )

        elif insertion_method == "prepend_prompt":
            original_prompt = req.prompt
            req.prompt = (
                f"{knowledge_to_insert}\n\n"
                f"{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
            )
            log_success(
                "知识库内容注入",
                "prepend_prompt",
                details={
                    "原始长度": len(original_prompt),
                    "知识库长度": len(knowledge_to_insert),
                    "最终长度": len(req.prompt)
                }
            )

        else:
            log_warning(
                "知识库内容注入",
                f"未知插入方式: {insertion_method},使用默认方式",
                details={"默认方式": "prepend_prompt"}
            )
            req.prompt = (
                f"{knowledge_to_insert}\n\n"
                f"{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
            )

        # 记录修改后的内容（用于调试）
        log_debug(
            "LLM 请求内容",
            {"prompt_preview": req.prompt[:PROMPT_PREVIEW_LENGTH]}
        )
        if req.system_prompt:
            log_debug(
                "LLM 请求内容",
                {"system_prompt_preview": req.system_prompt[:PROMPT_PREVIEW_LENGTH]}
            )
