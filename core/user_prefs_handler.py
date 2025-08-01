# astrbot_plugin_knowledge_base/user_prefs_handler.py
import json
import os
from typing import Dict, AsyncGenerator, TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from ..config.settings import PluginSettings
if TYPE_CHECKING:
    from ..vector_store.base import VectorDBBase


class UserPrefsHandler:
    def __init__(
        self, prefs_path: str, vector_db: "VectorDBBase", config: PluginSettings
    ):
        self.user_prefs_path = prefs_path
        self.user_collection_preferences: Dict[str, str] = {}
        self.vector_db = vector_db
        self.settings = config

    async def load_user_preferences(self):
        try:
            if os.path.exists(self.user_prefs_path):
                with open(self.user_prefs_path, "r", encoding="utf-8") as f:
                    self.user_collection_preferences = json.load(f)
                logger.info(f"从 {self.user_prefs_path} 加载了用户知识库偏好。")
            else:
                logger.info(
                    f"用户知识库偏好文件 {self.user_prefs_path} 未找到，将使用默认值。"
                )
        except Exception as e:
            logger.error(f"加载用户知识库偏好失败: {e}")
            self.user_collection_preferences = {}

    async def save_user_preferences(self):
        try:
            with open(self.user_prefs_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.user_collection_preferences, f, ensure_ascii=False, indent=4
                )
            logger.info(f"用户知识库偏好已保存到 {self.user_prefs_path}。")
        except Exception as e:
            logger.error(f"保存用户知识库偏好失败: {e}")

    def get_user_default_collection(self, event: AstrMessageEvent) -> str:
        user_key = event.unified_msg_origin
        return self.user_collection_preferences.get(
            user_key, self.settings.default_collection_name
        )

    async def set_user_default_collection(
        self, event: AstrMessageEvent, collection_name: str
    ) -> AsyncGenerator[AstrMessageEvent, None]:
        if not await self.vector_db.collection_exists(collection_name):
            if self.settings.auto_create_collection:
                try:
                    await self.vector_db.create_collection(collection_name)
                    logger.info(f"自动创建知识库 '{collection_name}' 成功。")
                    yield event.plain_result(
                        f"自动创建知识库 '{collection_name}' 成功。"
                    )
                except Exception as e:
                    logger.error(f"自动创建知识库 '{collection_name}' 失败: {e}")
                    yield event.plain_result(
                        f"自动创建知识库 '{collection_name}' 失败: {e}"
                    )
                    return
            else:
                yield event.plain_result(
                    f"知识库 '{collection_name}' 不存在，且未配置自动创建。"
                )
                return

        user_key = event.unified_msg_origin
        self.user_collection_preferences[user_key] = collection_name
        await self.save_user_preferences()
        yield event.plain_result(f"当前会话默认知识库已设置为: {collection_name}")

    async def clear_user_collection_pref(self, event: AstrMessageEvent) -> None:
        """
        清除当前会话配置的默认知识库。
        """
        user_key = event.unified_msg_origin
        if user_key in self.user_collection_preferences:
            del self.user_collection_preferences[user_key]
            await self.save_user_preferences()

    def get_collection_name_by_file_id(self, file_id: str = None) -> dict:
        """获取集合的元数据，包括嵌入提供商信息"""
        metadatas = self.user_collection_preferences.get(
            "collection_metadata", {}
        )
        for collection_name, metadata in metadatas.items():
            if metadata.get("file_id") == file_id:
                return collection_name
        return None
