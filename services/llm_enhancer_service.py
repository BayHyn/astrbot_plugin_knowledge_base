from typing import TYPE_CHECKING

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

from ..core.constants import KB_START_MARKER, KB_END_MARKER, USER_PROMPT_DELIMITER_IN_HISTORY
from ..vector_store.base import VectorDBBase
from ..core.user_prefs_handler import UserPrefsHandler
from ..config.settings import PluginSettings

if TYPE_CHECKING:
    from .document_service import DocumentService


class LLMEnhancerService:
    def __init__(
        self,
        vector_db: VectorDBBase,
        user_prefs_handler: UserPrefsHandler,
        settings: PluginSettings,
    ):
        self.vector_db = vector_db
        self.user_prefs_handler = user_prefs_handler
        self.settings = settings

    async def enhance_request_with_kb(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        default_collection_name = self.user_prefs_handler.get_user_default_collection(event)

        if not default_collection_name:
            logger.debug("当前会话未找到默认知识库，跳过LLM请求增强。")
            return

        if not await self.vector_db.collection_exists(default_collection_name):
            logger.warning(f"用户的默认知识库 '{default_collection_name}' 不存在，跳过LLM请求增强。")
            return

        if not self.settings.enable_kb_llm_enhancement:
            logger.info("LLM请求的知识库增强功能已全局禁用。")
            return

        user_query = req.prompt
        if not user_query or not user_query.strip():
            logger.debug("用户查询为空，跳过知识库搜索。")
            return

        try:
            logger.info(f"在知识库 '{default_collection_name}' 中为LLM请求搜索: '{user_query[:50]}...' (top_k={self.settings.kb_llm_search_top_k})")
            search_results = await self.vector_db.search(
                default_collection_name, user_query, top_k=self.settings.kb_llm_search_top_k
            )
        except Exception as e:
            logger.error(f"为LLM请求搜索知识库 '{default_collection_name}' 失败: {e}", exc_info=True)
            return

        if not search_results:
            logger.info(f"在知识库 '{default_collection_name}' 中未找到与查询 '{user_query[:50]}...' 相关的内容。")
            return

        retrieved_contexts_list = []
        for doc, score in search_results:
            if score >= self.settings.kb_llm_min_similarity_score:
                source_info = doc.metadata.get("source", "未知来源")
                context_item = f"- 内容: {doc.text_content} (来源: {source_info}, 相关度: {score:.2f})"
                retrieved_contexts_list.append(context_item)
            else:
                logger.debug(f"文档 '{doc.text_content[:30]}...' 的相关度 {score:.2f} 低于阈值 {self.settings.kb_llm_min_similarity_score}，已忽略。")

        if not retrieved_contexts_list:
            logger.info(f"所有检索到的知识库内容都低于相关度阈值 {self.settings.kb_llm_min_similarity_score}，不执行增强。")
            return

        formatted_contexts = "\n".join(retrieved_contexts_list)
        kb_context_template = self.settings.kb_llm_context_template
        knowledge_to_insert = kb_context_template.format(retrieved_contexts=formatted_contexts)
        
        knowledge_to_insert = f"{KB_START_MARKER}\n{knowledge_to_insert}\n{KB_END_MARKER}"

        if self.settings.kb_llm_insertion_method == "system_prompt":
            if req.system_prompt:
                req.system_prompt = f"{knowledge_to_insert}\n\n{req.system_prompt}"
            else:
                req.system_prompt = knowledge_to_insert
            logger.info(f"知识库内容已添加到 system_prompt。长度: {len(knowledge_to_insert)}")
        else: # prepend_prompt 是默认值
            req.prompt = f"{knowledge_to_insert}\n\n{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
            logger.info(f"知识库内容已前置到用户提示。长度: {len(knowledge_to_insert)}")


def clean_contexts_from_kb_content(req: ProviderRequest):
    """
    自动从 req.contexts 的历史记录中删除由知识库添加的内容。
    """
    if not req.contexts:
        return

    cleaned_contexts = []
    initial_context_count = len(req.contexts)

    for message in req.contexts:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system" and KB_START_MARKER in content:
            logger.debug(f"检测到并从历史记录中删除知识库系统消息: {content[:100]}...")
            continue
        elif role == "user" and KB_START_MARKER in content:
            start_marker_idx = content.find(KB_START_MARKER)
            end_marker_idx = content.find(KB_END_MARKER, start_marker_idx)
            if start_marker_idx != -1 and end_marker_idx != -1:
                original_prompt_delimiter_idx = content.find(
                    USER_PROMPT_DELIMITER_IN_HISTORY,
                    end_marker_idx + len(KB_END_MARKER),
                )
                if original_prompt_delimiter_idx != -1:
                    original_user_prompt = content[
                        original_prompt_delimiter_idx
                        + len(USER_PROMPT_DELIMITER_IN_HISTORY) :
                    ].strip()
                    message["content"] = original_user_prompt
                    cleaned_contexts.append(message)
                    logger.debug(f"从历史记录中的用户消息中清除了知识库内容，保留原始用户问题: {original_user_prompt[:100]}...")
                else:
                    logger.warning(f"在用户消息中检测到知识库标记，但缺少原始用户问题分隔符，正在删除消息: {content[:100]}...")
                    continue
            else:
                logger.warning(f"在用户消息中检测到知识库开始标记，但缺少结束标记，正在删除消息: {content[:100]}...")
                continue
        else:
            cleaned_contexts.append(message)

    req.contexts = cleaned_contexts
    if len(req.contexts) < initial_context_count:
        logger.info(f"成功从历史记录中删除 {initial_context_count - len(req.contexts)} 条知识库补充消息。")