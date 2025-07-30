from typing import Optional, TYPE_CHECKING, AsyncGenerator
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin


async def handle_add_text(
    plugin: "KnowledgeBasePlugin",
    event: AstrMessageEvent,
    content: str,
    collection_name: Optional[str] = None,
) -> AsyncGenerator[AstrMessageEvent, None]:
    """处理向知识库添加文本的命令。"""
    target_collection = (
        collection_name or plugin.user_prefs_handler.get_user_default_collection(event)
    )

    async for result_event in plugin.document_service.add_text_to_collection(
        content, target_collection, event
    ):
        yield result_event


async def handle_add_file(
    plugin: "KnowledgeBasePlugin",
    event: AstrMessageEvent,
    path_or_url: str,
    collection_name: Optional[str] = None,
) -> AsyncGenerator[AstrMessageEvent, None]:
    """处理向知识库添加文件或URL内容的命令。"""
    target_collection = (
        collection_name or plugin.user_prefs_handler.get_user_default_collection(event)
    )

    async for result_event in plugin.document_service.add_file_to_collection(
        path_or_url, target_collection, event
    ):
        yield result_event
