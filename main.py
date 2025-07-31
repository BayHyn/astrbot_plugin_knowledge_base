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
        """初始化组件，增强容错性"""
        logger.info("知识库插件开始初始化...")
        
        # 初始化状态跟踪
        initialization_steps = [
            "embedding_util",
            "user_prefs_handler",
            "vector_db", 
            "kb_service",
            "document_service", 
            "llm_enhancer_service",
            "web_api"
        ]
        
        failed_steps = []
        
        try:
            # 1. 初始化嵌入工具 - 直接使用AstrBot内置的提供商
            try:
                from .utils.embedding import EmbeddingUtil
                embedding_helper = EmbeddingUtil(provider_context=self.context)
                logger.info("✓ 嵌入工具初始化成功")
            except Exception as e:
                failed_steps.append("embedding_util")
                logger.error(f"✗ 嵌入工具初始化失败: {e}")
                raise

            # 2. 初始化用户偏好处理器（需要在向量数据库之前初始化）
            try:
                self.user_prefs_handler = UserPrefsHandler(
                    prefs_path=self.user_prefs_path,
                    vector_db=None,  # 先传None，后面会设置
                    config=self.plugin_config,
                )
                await self.user_prefs_handler.load_user_preferences()
                logger.info("✓ 用户偏好处理器初始化成功")
            except Exception as e:
                failed_steps.append("user_prefs_handler")
                logger.error(f"✗ 用户偏好处理器初始化失败: {e}")
                # 用户偏好处理器失败不应该阻止插件启动
                logger.warning("用户偏好处理器初始化失败，将创建默认实例")
                self.user_prefs_handler = None

            # 3. 初始化向量数据库
            try:
                from .vector_store.enhanced_faiss_store import EnhancedFaissStore
                vector_db = EnhancedFaissStore(
                    embedding_util=embedding_helper,
                    data_path=self.persistent_data_root_path,
                    user_prefs_handler=self.user_prefs_handler,
                )
                await vector_db.initialize()
                
                # 设置向量数据库到用户偏好处理器
                if self.user_prefs_handler:
                    self.user_prefs_handler.vector_db = vector_db
                    
                logger.info("✓ 向量数据库初始化成功")
            except Exception as e:
                failed_steps.append("vector_db")
                logger.error(f"✗ 向量数据库初始化失败: {e}")
                raise

            # 4. 初始化知识库服务
            try:
                self.kb_service = KnowledgeBaseService(
                    vector_db=vector_db,
                    user_prefs_handler=self.user_prefs_handler,
                    settings=self.plugin_config,
                )
                logger.info("✓ 知识库服务初始化成功")
            except Exception as e:
                failed_steps.append("kb_service")
                logger.error(f"✗ 知识库服务初始化失败: {e}")
                raise

            # 5. 初始化文档服务
            try:
                from .utils.file_parser import FileParser
                from .utils.text_splitter import TextSplitterUtil

                file_parser = FileParser(self.context, self.plugin_config.llm_parser)
                text_splitter = TextSplitterUtil(
                    chunk_size=self.plugin_config.text_chunk_size,
                    chunk_overlap=self.plugin_config.text_chunk_overlap
                )
                self.document_service = DocumentService(
                    vector_db=vector_db,
                    kb_service=self.kb_service,
                    file_parser=file_parser,
                    text_splitter=text_splitter,
                    data_path=self.persistent_data_root_path,
                )
                logger.info("✓ 文档服务初始化成功")
            except Exception as e:
                failed_steps.append("document_service")
                logger.error(f"✗ 文档服务初始化失败: {e}")
                raise

            # 6. 初始化LLM增强服务
            try:
                self.llm_enhancer_service = LLMEnhancerService(
                    vector_db=vector_db,
                    user_prefs_handler=self.user_prefs_handler,
                    settings=self.plugin_config,
                )
                logger.info("✓ LLM增强服务初始化成功")
            except Exception as e:
                failed_steps.append("llm_enhancer_service")
                logger.error(f"✗ LLM增强服务初始化失败: {e}")
                # LLM增强服务失败不应该阻止插件启动
                logger.warning("LLM增强服务初始化失败，RAG功能将不可用")
                self.llm_enhancer_service = None

            # 7. 初始化Web API
            try:
                self.web_api = KnowledgeBaseWebAPI(
                    kb_service=self.kb_service,
                    document_service=self.document_service,
                    astrbot_context=self.context,
                    plugin_config=self.plugin_config,
                )
                logger.info("✓ WebAPI初始化成功")
            except Exception as e:
                failed_steps.append("web_api")
                logger.warning(f"✗ WebAPI初始化失败: {e}")
                logger.warning("WebAPI初始化失败，WebUI知识库管理功能将不可用")
                self.web_api = None

            # 输出初始化结果总结
            success_count = len(initialization_steps) - len(failed_steps)
            logger.info(f"知识库插件初始化完成: {success_count}/{len(initialization_steps)} 个组件成功初始化")
            
            if failed_steps:
                logger.warning(f"以下组件初始化失败: {', '.join(failed_steps)}")
                # 如果核心组件失败，不启动插件
                critical_components = ["embedding_util", "user_prefs_handler", "vector_db", "kb_service", "document_service"]
                failed_critical = [step for step in failed_steps if step in critical_components]
                if failed_critical:
                    logger.error(f"关键组件初始化失败: {', '.join(failed_critical)}")
                    raise Exception(f"关键组件初始化失败: {', '.join(failed_critical)}")

        except Exception as e:
            logger.error(f"知识库插件初始化失败: {e}", exc_info=True)
            # 清理已初始化的组件
            await self._cleanup_on_failure()
            self.kb_service = None
            raise

    async def _cleanup_on_failure(self):
        """初始化失败时的清理工作"""
        try:
            if hasattr(self, 'kb_service') and self.kb_service:
                await self.kb_service.close()
                logger.info("已清理知识库服务")
        except Exception as e:
            logger.warning(f"清理知识库服务时出错: {e}")
        
        try:
            if hasattr(self, 'user_prefs_handler') and self.user_prefs_handler:
                await self.user_prefs_handler.save_user_preferences()
                logger.info("已保存用户偏好设置")
        except Exception as e:
            logger.warning(f"保存用户偏好设置时出错: {e}")

    async def _ensure_initialized(self) -> bool:
        """确保插件已正确初始化，增强容错性"""
        try:
            if self.init_task and not self.init_task.done():
                await asyncio.wait_for(self.init_task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("插件初始化超时")
            return False
        except Exception as e:
            logger.error(f"等待插件初始化时出错: {e}")
            return False
            
        # 检查核心组件是否可用
        if not self.kb_service:
            logger.error("知识库服务未正确初始化")
            return False
            
        if not self.document_service:
            logger.error("文档服务未正确初始化")
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
