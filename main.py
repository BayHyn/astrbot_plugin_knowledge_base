# astrbot_plugin_knowledge_base/main.py
import os
import asyncio
from typing import Optional

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.session_waiter import (
    session_waiter,
    SessionController,
)
from astrbot.core.config.default import VERSION
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import StarTools

from .config.settings import PluginSettings
from .services.document_service import DocumentService
from .services.kb_service import KnowledgeBaseService
from .services.llm_enhancer_service import LLMEnhancerService
from .core.user_prefs_handler import UserPrefsHandler
from .web_api import KnowledgeBaseWebAPI
from .commands import general_commands


@register(
    "astrbot_plugin_knowledge_base",
    "lxfight",
    "一个支持多种向量数据库的知识库插件",
    "0.5.4",
    "https://github.com/lxfight/astrbot_plugin_knowledge_base",
)
class KnowledgeBasePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._initialize_basic_paths()

        # Services
        self.kb_service: Optional[KnowledgeBaseService] = None
        self.document_service: Optional[DocumentService] = None
        self.llm_enhancer_service: Optional[LLMEnhancerService] = None
        self.user_prefs_handler: Optional[UserPrefsHandler] = None

        # Initialize plugin config
        self.plugin_config = PluginSettings.from_astrbot_config(config)

        # Start initialization
        self.init_task = asyncio.create_task(self._initialize_components())

    def _initialize_basic_paths(self):
        self.plugin_name_for_path = "astrbot_plugin_knowledge_base"
        self.persistent_data_root_path = StarTools.get_data_dir(
            self.plugin_name_for_path
        )
        os.makedirs(self.persistent_data_root_path, exist_ok=True)
        logger.info(f"知识库插件的持久化数据目录: {self.persistent_data_root_path}")
        self.user_prefs_path = os.path.join(
            self.persistent_data_root_path, "user_collection_prefs.json"
        )

    async def _initialize_components(self):
        try:
            logger.info("知识库插件开始初始化...")

            # --- 依赖注入顺序 ---
            # 1. 初始化嵌入工具
            from .utils.embedding import EmbeddingUtil

            embedding_helper = EmbeddingUtil(self.plugin_config.embedding)

            # 2. 初始化向量数据库
            from .vector_store.enhanced_faiss_store import EnhancedFaissStore

            vector_db = EnhancedFaissStore(
                embedding_util=embedding_helper,
                data_path=self.persistent_data_root_path,
            )
            await vector_db.initialize()

            # 3. 初始化用户偏好处理器
            self.user_prefs_handler = UserPrefsHandler(
                prefs_path=self.user_prefs_path,
                vector_db=vector_db,
                config=self.plugin_config,
            )
            await self.user_prefs_handler.load_user_preferences()

            # 4. 初始化知识库服务
            self.kb_service = KnowledgeBaseService(
                vector_db=vector_db,
                user_prefs_handler=self.user_prefs_handler,
                settings=self.plugin_config,
            )

            # 5. 初始化文档服务
            from .utils.file_parser import FileParser
            from .utils.text_splitter import TextSplitterUtil

            file_parser = FileParser(self.context, self.plugin_config.llm_parser)
            text_splitter = TextSplitterUtil(chunk_size=1000, chunk_overlap=200)
            self.document_service = DocumentService(
                vector_db=vector_db,
                kb_service=self.kb_service,
                file_parser=file_parser,
                text_splitter=text_splitter,
                data_path=self.persistent_data_root_path,
            )

            # 6. 初始化LLM增强服务
            self.llm_enhancer_service = LLMEnhancerService(
                vector_db=vector_db,
                user_prefs_handler=self.user_prefs_handler,
                settings=self.plugin_config,
            )

            # 7. Initialize Web API
            try:
                self.web_api = KnowledgeBaseWebAPI(
                    kb_service=self.kb_service,
                    document_service=self.document_service,
                    astrbot_context=self.context,
                    plugin_config=self.plugin_config,
                )
            except Exception as e:
                logger.warning(
                    f"知识库 WebAPI 初始化失败，可能导致无法在 WebUI 操作知识库。原因：{e}",
                    exc_info=True,
                )

            logger.info("知识库插件初始化成功。")

        except Exception as e:
            logger.error(f"知识库插件初始化失败: {e}", exc_info=True)
            self.kb_service = None

    async def _ensure_initialized(self) -> bool:
        if self.init_task and not self.init_task.done():
            await self.init_task
        if not self.kb_service:
            logger.error("知识库插件未正确初始化，请检查日志和配置。")
            return False
        return True

    # --- LLM Request Hook ---
    @filter.on_llm_request()
    async def kb_on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not await self._ensure_initialized():
            logger.warning("LLM 请求时知识库插件未初始化，跳过知识库增强。")
            return

        await self.llm_enhancer_service.enhance_request(
            event, req, self.user_prefs_handler
        )

    # --- Command Groups & Commands ---
    @filter.command_group("kb", alias={"knowledge", "知识库"})
    def kb_group(self):
        """知识库管理指令集"""
        pass

    @kb_group.command("help", alias={"帮助"})
    async def kb_help(self, event: AstrMessageEvent):
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_help(self, event):
            yield result

    @kb_group.command("current", alias={"当前"})
    async def kb_current_collection(self, event: AstrMessageEvent):
        """查看当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_current_collection(self, event):
            yield result

    @kb_group.command("use", alias={"使用", "set"})
    async def kb_use_collection(self, event: AstrMessageEvent, collection_name: str):
        """设置当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_use_collection(
            self, event, collection_name
        ):
            yield result

    @kb_group.command("clear_use")
    async def kb_clear_use_collection(self, event: AstrMessageEvent):
        """清除默认使用的知识库，并关闭RAG知识库补充功能"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        try:
            await self.user_prefs_handler.clear_user_collection_pref(event)
            yield event.plain_result("已清除默认知识库，并关闭RAG知识库补充功能。")
        except Exception as e:
            logger.error(f"清除默认知识库时发生错误: {e}", exc_info=True)
            yield event.plain_result(f"清除默认知识库失败: {e}")

    # --- Termination ---
    async def terminate(self):
        logger.info("知识库插件正在终止...")
        if hasattr(self, "init_task") and self.init_task and not self.init_task.done():
            logger.info("等待初始化任务完成...")
            try:
                await asyncio.wait_for(self.init_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("初始化任务超时，尝试取消。")
                self.init_task.cancel()
            except Exception as e:
                logger.error(f"等待初始化任务完成时出错: {e}")

        if self.kb_service:
            await self.kb_service.close()
            logger.info("知识库服务已关闭。")

        if self.user_prefs_handler:
            await self.user_prefs_handler.save_user_preferences()

        logger.info("知识库插件终止完成。")
