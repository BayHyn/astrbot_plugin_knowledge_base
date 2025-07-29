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
            logger.debug("No default knowledge base found for the current session, skipping LLM request enhancement.")
            return

        if not await self.vector_db.collection_exists(default_collection_name):
            logger.warning(f"User's default knowledge base '{default_collection_name}' does not exist, skipping LLM request enhancement.")
            return

        if not self.settings.enable_kb_llm_enhancement:
            logger.info("Knowledge base enhancement for LLM requests is globally disabled.")
            return

        user_query = req.prompt
        if not user_query or not user_query.strip():
            logger.debug("User query is empty, skipping knowledge base search.")
            return

        try:
            logger.info(f"Searching in knowledge base '{default_collection_name}' for LLM request: '{user_query[:50]}...' (top_k={self.settings.kb_llm_search_top_k})")
            search_results = await self.vector_db.search(
                default_collection_name, user_query, top_k=self.settings.kb_llm_search_top_k
            )
        except Exception as e:
            logger.error(f"Failed to search knowledge base '{default_collection_name}' for LLM request: {e}", exc_info=True)
            return

        if not search_results:
            logger.info(f"No relevant content found in knowledge base '{default_collection_name}' for query '{user_query[:50]}...'.")
            return

        retrieved_contexts_list = []
        for doc, score in search_results:
            if score >= self.settings.kb_llm_min_similarity_score:
                source_info = doc.metadata.get("source", "Unknown source")
                context_item = f"- Content: {doc.text_content} (Source: {source_info}, Relevance: {score:.2f})"
                retrieved_contexts_list.append(context_item)
            else:
                logger.debug(f"Document '{doc.text_content[:30]}...' with relevance {score:.2f} is below the threshold {self.settings.kb_llm_min_similarity_score}, ignored.")

        if not retrieved_contexts_list:
            logger.info(f"All retrieved knowledge base content is below the relevance threshold {self.settings.kb_llm_min_similarity_score}, no enhancement will be performed.")
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
            logger.info(f"Knowledge base content has been added to system_prompt. Length: {len(knowledge_to_insert)}")
        else: # prepend_prompt is the default
            req.prompt = f"{knowledge_to_insert}\n\n{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
            logger.info(f"Knowledge base content has been prepended to the user prompt. Length: {len(knowledge_to_insert)}")


def clean_contexts_from_kb_content(req: ProviderRequest):
    """
    Automatically removes content added by the knowledge base from the history of the req.contexts.
    """
    if not req.contexts:
        return

    cleaned_contexts = []
    initial_context_count = len(req.contexts)

    for message in req.contexts:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system" and KB_START_MARKER in content:
            logger.debug(f"Detected and removed knowledge base system message from history: {content[:100]}...")
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
                    logger.debug(f"Cleaned knowledge base content from user message in history, retaining original user question: {original_user_prompt[:100]}...")
                else:
                    logger.warning(f"Detected knowledge base marker in user message but missing original user question delimiter, removing the message: {content[:100]}...")
                    continue
            else:
                logger.warning(f"Detected knowledge base start marker in user message but missing end marker, removing the message: {content[:100]}...")
                continue
        else:
            cleaned_contexts.append(message)

    req.contexts = cleaned_contexts
    if len(req.contexts) < initial_context_count:
        logger.info(f"Successfully removed {initial_context_count - len(req.contexts)} knowledge base supplement messages from history.")