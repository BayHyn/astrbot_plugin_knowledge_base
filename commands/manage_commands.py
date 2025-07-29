# astrbot_plugin_knowledge_base/commands/manage_commands.py
from typing import Optional, TYPE_CHECKING, AsyncGenerator
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin


async def handle_list_collections(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent
) -> AsyncGenerator[AstrMessageEvent, None]:
    try:
        collections = await plugin.kb_service.list_collections()
        if not collections:
            yield event.plain_result("当前没有可用的知识库。")
            return

        response = "可用的知识库列表:\n"
        for col_name, count in collections.items():
            response += f"- {col_name} (文档数: {count})\n"
        yield event.plain_result(response.strip())
    except Exception as e:
        logger.error(f"列出知识库失败: {e}", exc_info=True)
        yield event.plain_result(f"列出知识库失败: {e}")


async def handle_create_collection(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent, collection_name: str
) -> AsyncGenerator[AstrMessageEvent, None]:
    if not collection_name:
        yield event.plain_result(
            "请输入要创建的知识库名称。用法: /kb create <知识库名>"
        )
        return

    try:
        await plugin.kb_service.create_collection(collection_name)
        yield event.plain_result(f"知识库 '{collection_name}' 创建成功。")
    except ValueError as e:
        yield event.plain_result(str(e))
    except Exception as e:
        logger.error(f"创建知识库 '{collection_name}' 失败: {e}", exc_info=True)
        yield event.plain_result(f"创建知识库 '{collection_name}' 失败: {e}")


async def handle_delete_collection_logic(
    plugin: "KnowledgeBasePlugin", confirm_event: AstrMessageEvent, collection_name: str
):
    """由会话等待器调用的实际删除逻辑。"""
    try:
        await confirm_event.send(
            confirm_event.plain_result(f"正在删除知识库 '{collection_name}'...")
        )
        await plugin.kb_service.delete_collection(collection_name)
        await confirm_event.send(
            confirm_event.plain_result(f"知识库 '{collection_name}' 已成功删除。")
        )
    except ValueError as e:
        await confirm_event.send(
            confirm_event.plain_result(str(e))
        )
    except Exception as e_del:
        logger.error(
            f"删除知识库 '{collection_name}' 过程中发生错误: {e_del}", exc_info=True
        )
        await confirm_event.send(
            confirm_event.plain_result(f"删除知识库 '{collection_name}' 失败: {e_del}")
        )


async def handle_count_documents(
    plugin: "KnowledgeBasePlugin",
    event: AstrMessageEvent,
    collection_name: Optional[str] = None,
) -> AsyncGenerator[AstrMessageEvent, None]:
    target_collection = (
        collection_name
        if collection_name
        else plugin.user_prefs_handler.get_user_default_collection(event)
    )

    try:
        count = await plugin.kb_service.count_documents(target_collection)
        yield event.plain_result(
            f"知识库 '{target_collection}' 中包含 {count} 个文档块。"
        )
    except ValueError as e:
        yield event.plain_result(str(e))
    except Exception as e:
        logger.error(
            f"获取知识库 '{target_collection}' 文档数量失败: {e}", exc_info=True
        )
        yield event.plain_result(f"获取文档数量失败: {e}")


async def handle_migrate_files(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent, faiss_path: str
):
    try:
        from ..utils.migrate_files import migrate_docs_to_db
        migrate_docs_to_db(faiss_path)
    except Exception as e:
        raise Exception(f"迁移文件失败，请检查日志。{e}", exc_info=True)
