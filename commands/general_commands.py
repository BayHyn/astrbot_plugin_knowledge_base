# astrbot_plugin_knowledge_base/commands/general_commands.py
from typing import TYPE_CHECKING, AsyncGenerator
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin


async def handle_kb_help(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent
) -> AsyncGenerator[AstrMessageEvent, None]:
    help_text = """
知识库插件帮助：

📋 用户偏好命令：
/kb current - 查看当前会话默认知识库
/kb use <知识库名> - 设置当前会话默认知识库
/kb clear_use - 清除默认知识库设置，关闭RAG功能
/kb help - 显示此帮助信息

🌐 知识库管理：
请使用 WebUI 进行知识库的创建、删除、添加文档、搜索等操作。

💡 说明：
- 设置默认知识库后，对话时会自动使用该知识库进行RAG增强
- 所有知识库管理功能都可通过Web界面操作，更加便捷
""".strip()
    yield event.plain_result(help_text)


async def handle_kb_current_collection(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent
) -> AsyncGenerator[AstrMessageEvent, None]:
    current_col = plugin.user_prefs_handler.get_user_default_collection(event)
    yield event.plain_result(f"当前会话默认知识库为: {current_col}")


async def handle_kb_use_collection(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent, collection_name: str
) -> AsyncGenerator[AstrMessageEvent, None]:
    if not collection_name:
        yield event.plain_result("请输入要设置的知识库名称。用法: /kb use <知识库名>")
        return

    try:
        async for result in plugin.user_prefs_handler.set_user_default_collection(event, collection_name):
            yield result
    except Exception as e:
        yield event.plain_result(f"设置默认知识库失败: {e}")
