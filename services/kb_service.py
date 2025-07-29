from typing import List, Optional, AsyncGenerator

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..vector_store.base import VectorDBBase
from ..core.user_prefs_handler import UserPrefsHandler
from ..config.settings import PluginSettings


class KnowledgeBaseService:
    def __init__(
        self,
        vector_db: VectorDBBase,
        user_prefs_handler: UserPrefsHandler,
        settings: PluginSettings,
    ):
        self.vector_db = vector_db
        self.user_prefs_handler = user_prefs_handler
        self.settings = settings

    async def list_collections(self, event: AstrMessageEvent) -> AsyncGenerator[AstrMessageEvent, None]:
        try:
            collections = await self.vector_db.list_collections()
            if not collections:
                yield event.plain_result("当前没有可用的知识库。")
                return

            response = "可用的知识库列表:\n"
            for col_name in collections:
                count = await self.vector_db.count_documents(col_name)
                response += f"- {col_name} (文档数: {count})\n"
            yield event.plain_result(response.strip())
        except Exception as e:
            logger.error(f"列出知识库失败: {e}", exc_info=True)
            yield event.plain_result(f"列出知识库失败: {e}")

    async def create_collection(self, collection_name: str, event: AstrMessageEvent) -> AsyncGenerator[AstrMessageEvent, None]:
        if not collection_name:
            yield event.plain_result(
                "请输入要创建的知识库名称。用法: /kb create <知识库名>"
            )
            return

        if await self.vector_db.collection_exists(collection_name):
            yield event.plain_result(f"知识库 '{collection_name}' 已存在。")
            return

        try:
            await self.vector_db.create_collection(collection_name)
            yield event.plain_result(f"知识库 '{collection_name}' 创建成功。")
        except Exception as e:
            logger.error(f"创建知识库 '{collection_name}' 失败: {e}", exc_info=True)
            yield event.plain_result(f"创建知识库 '{collection_name}' 失败: {e}")

    async def ensure_collection_exists(self, collection_name: str, event: AstrMessageEvent):
        if not await self.vector_db.collection_exists(collection_name):
            if self.settings.auto_create_collection:
                try:
                    await self.vector_db.create_collection(collection_name)
                    logger.info(f"知识库 '{collection_name}' 不存在，已自动创建。")
                    await event.send(event.plain_result(
                        f"知识库 '{collection_name}' 不存在，已自动创建。"
                    ))
                except Exception as e:
                    logger.error(f"自动创建知识库 '{collection_name}' 失败: {e}")
                    await event.send(event.plain_result(
                        f"自动创建知识库 '{collection_name}' 失败: {e}"
                    ))
                    raise e
            else:
                raise ValueError(f"知识库 '{collection_name}' 不存在，且自动创建功能已禁用。")