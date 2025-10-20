# astrbot_plugin_knowledge_base/llm_enhancer.py
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

if TYPE_CHECKING:
    from ..vector_store.base import VectorDBBase
    from .user_prefs_handler import UserPrefsHandler
    from .config_manager import ConfigManager
    from .services import RAGService


async def enhance_request_with_kb(
    event: AstrMessageEvent,
    req: ProviderRequest,
    rag_service: "RAGService",
    user_prefs_handler: "UserPrefsHandler",
):
    """使用知识库增强 LLM 请求 (兼容层函数)

    这是一个兼容层函数,将旧的调用方式转换为新的服务层调用。

    Args:
        event: 消息事件
        req: LLM 请求对象
        rag_service: RAG 服务实例
        user_prefs_handler: 用户偏好处理器
    """
    # 清理历史对话中的知识库内容
    rag_service.clean_kb_content_from_contexts(req)

    # 获取默认知识库
    default_collection_name = user_prefs_handler.get_user_default_collection(event)

    # 明确检查 None 和空字符串
    if default_collection_name is None or default_collection_name == "":
        logger.debug("未设置默认知识库（返回值为 None 或空字符串），跳过知识库查询。")
        return

    # 使用 RAG 服务增强请求
    try:
        await rag_service.enhance_request(event, req, default_collection_name)
    except ValueError as e:
        logger.warning(f"知识库增强失败: {e}")
    except Exception as e:
        logger.error(f"知识库增强时发生未预期的错误: {e}", exc_info=True)
