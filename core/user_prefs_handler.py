# astrbot_plugin_knowledge_base/user_prefs_handler.py
import json
import os
from typing import Dict, AsyncGenerator, TYPE_CHECKING, Optional

from astrbot.api import logger, AstrBotConfig
from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent
from astrbot.core.config.default import VERSION

if TYPE_CHECKING:
    from ..vector_store.base import VectorDBBase


class UserPrefsHandler:
    def __init__(
        self,
        prefs_path: str,
        config: AstrBotConfig,
        context: Context,
    ):
        """初始化用户偏好处理器，使用延迟注入模式处理 vector_db 依赖"""
        self.user_prefs_path = prefs_path
        self.user_collection_preferences: Dict[str, str] = {}
        self._vector_db: Optional["VectorDBBase"] = None
        self.config = config
        self.context = context

    def set_vector_db(self, vector_db: "VectorDBBase") -> None:
        """延迟注入 vector_db 依赖"""
        self._vector_db = vector_db
        logger.debug("VectorDB 已成功注入到 UserPrefsHandler")

    @property
    def vector_db(self) -> "VectorDBBase":
        """获取 vector_db 实例，如果未初始化则抛出异常"""
        if self._vector_db is None:
            raise RuntimeError(
                "VectorDB 尚未初始化。请确保在使用前调用 set_vector_db() 方法。"
            )
        return self._vector_db

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

    def get_user_default_collection(self, event: AstrMessageEvent) -> Optional[str]:
        """
        获取用户的默认知识库集合名称

        返回值说明:
        - str: 知识库集合名称（非空字符串）
        - None: 未设置知识库或显式设置为不使用知识库

        优先级：
        1. 用户会话偏好（user_collection_preferences）
        2. AstrBot 4.0+ 配置（default_kb_collection）
        3. 插件配置（default_collection_name，仅 AstrBot < 4.0.0）
        """
        user_key = event.unified_msg_origin

        # 优先级1: 用户会话偏好
        if user_kb_perf := self.user_collection_preferences.get(user_key, None):
            return user_kb_perf

        # 优先级2: AstrBot 4.0+ 配置
        if VERSION >= "4.0.0":
            astrbot_cfg = self.context.get_config(umo=user_key)
            collection_name = astrbot_cfg.get("default_kb_collection", None)
            # 将空字符串统一转换为 None，表示未设置知识库
            if collection_name == "":
                return None
            return collection_name

        # 优先级3: 插件配置（< 4.0.0 版本）
        default_name = self.config.get("default_collection_name", None)
        # 将空字符串和 "general" 默认值转换为 None
        if default_name == "" or default_name is None:
            return None
        return default_name


    async def set_user_default_collection(
        self, event: AstrMessageEvent, collection_name: str
    ) -> AsyncGenerator[AstrMessageEvent, None]:
        if not await self.vector_db.collection_exists(collection_name):
            if self.config.get("auto_create_collection", True):
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
        metadatas = self.user_collection_preferences.get("collection_metadata", {})
        for collection_name, metadata in metadatas.items():
            if metadata.get("file_id") == file_id:
                return collection_name
        return None
