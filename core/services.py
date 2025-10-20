# astrbot_plugin_knowledge_base/core/services.py
"""服务层 - 封装核心业务逻辑"""
from typing import List, Tuple, Optional, TYPE_CHECKING
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from .constants import KB_START_MARKER, KB_END_MARKER, USER_PROMPT_DELIMITER_IN_HISTORY
from .domain import CollectionMetadataRepository

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

        logger.debug(
            f"搜索参数: collection='{collection_name}', query='{query[:50]}...', "
            f"top_k={top_k}, min_similarity={min_similarity}"
        )

        # 执行搜索
        try:
            results = await self.vector_db.search(collection_name, query, top_k=top_k)
            logger.info(
                f"从知识库 '{collection_name}' 检索到 {len(results)} 个文档"
            )
        except Exception as e:
            logger.error(f"搜索失败: {e}", exc_info=True)
            raise

        # 过滤低相似度结果
        filtered_results = [
            (doc, score) for doc, score in results if score >= min_similarity
        ]

        if len(filtered_results) < len(results):
            logger.debug(
                f"过滤后保留 {len(filtered_results)}/{len(results)} 个文档 "
                f"(相似度阈值: {min_similarity})"
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
                logger.debug(
                    f"从历史对话中删除知识库 system 消息: {content[:100]}..."
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
                        logger.debug(
                            f"从历史对话 user 消息中清理知识库内容,保留原用户问题: "
                            f"{original_prompt[:100]}..."
                        )
                    else:
                        logger.warning(
                            f"用户消息中检测到知识库标记但缺少原始问题分隔符,删除该消息: "
                            f"{content[:100]}..."
                        )
                        continue
                else:
                    logger.warning(
                        f"用户消息中检测到知识库起始标记但缺少结束标记,删除该消息: "
                        f"{content[:100]}..."
                    )
                    continue
            else:
                # 其他消息保持不变
                cleaned_contexts.append(message)

        req.contexts = cleaned_contexts
        removed_count = initial_count - len(req.contexts)

        if removed_count > 0:
            logger.info(f"成功从历史对话中删除了 {removed_count} 条知识库补充消息")

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
            logger.debug("用户查询为空,跳过知识库增强")
            return

        # 搜索相关知识
        try:
            logger.info(
                f"为 LLM 请求在知识库 '{collection_name}' 中搜索: "
                f"'{user_query[:50]}...' (top_k={user_config.search_top_k})"
            )
            search_results = await self.search_service.search(
                collection_name=collection_name,
                query=user_query,
                top_k=user_config.search_top_k,
                min_similarity=user_config.min_similarity_score,
            )
        except Exception as e:
            logger.error(
                f"LLM 请求时从知识库 '{collection_name}' 搜索失败: {e}",
                exc_info=True,
            )
            return

        if not search_results:
            logger.info(
                f"在知识库 '{collection_name}' 中未找到相关内容 "
                f"(查询: '{user_query[:50]}...')"
            )
            return

        logger.info(
            f"从知识库检索到 {len(search_results)} 个相关文档,准备注入到 LLM 请求"
        )

        # 格式化检索内容
        retrieved_contexts_list = []
        for idx, (doc, score) in enumerate(search_results):
            source_info = doc.metadata.get("source", "未知来源")
            context_item = (
                f"- 内容: {doc.text_content} (来源: {source_info}, 相关度: {score:.2f})"
            )
            retrieved_contexts_list.append(context_item)
            logger.debug(
                f"文档 #{idx+1}: 相关度={score:.4f}, 来源={source_info}, "
                f"内容='{doc.text_content[:50]}...'"
            )

        formatted_contexts = "\n".join(retrieved_contexts_list)
        knowledge_to_insert = user_config.context_template.format(
            retrieved_contexts=formatted_contexts
        )

        # 检查长度限制
        max_length = user_config.max_insert_length
        original_length = len(knowledge_to_insert)

        if original_length > max_length:
            logger.warning(
                f"知识库插入内容过长 ({original_length} chars), "
                f"将被截断至 {max_length} chars"
            )
            knowledge_to_insert = (
                knowledge_to_insert[:max_length] + "\n... [内容已截断]"
            )
        else:
            logger.debug(
                f"知识库内容长度: {original_length} chars "
                f"(未超过限制 {max_length})"
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
            logger.info(
                f"知识库内容已添加到 system_prompt "
                f"(最终长度: {len(req.system_prompt)} chars, "
                f"知识库部分: {len(knowledge_to_insert)} chars)"
            )

        elif insertion_method == "prepend_prompt":
            original_prompt = req.prompt
            req.prompt = (
                f"{knowledge_to_insert}\n\n"
                f"{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
            )
            logger.info(
                f"知识库内容已前置到用户 prompt "
                f"(原始长度: {len(original_prompt)} chars, "
                f"知识库长度: {len(knowledge_to_insert)} chars, "
                f"最终长度: {len(req.prompt)} chars)"
            )

        else:
            logger.warning(
                f"未知的知识库内容插入方式: {insertion_method}, "
                f"将默认前置到用户 prompt"
            )
            req.prompt = (
                f"{knowledge_to_insert}\n\n"
                f"{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
            )

        # 记录修改后的内容（用于调试）
        logger.debug(
            f"修改后的 ProviderRequest.prompt (前200字符): {req.prompt[:200]}..."
        )
        if req.system_prompt:
            logger.debug(
                f"修改后的 ProviderRequest.system_prompt (前200字符): "
                f"{req.system_prompt[:200]}..."
            )
