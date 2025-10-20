# astrbot_plugin_knowledge_base/commands/base.py
"""命令处理基类 - 统一命令处理模式"""
from typing import Optional, TYPE_CHECKING, AsyncGenerator
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin


class CommandContext:
    """命令执行上下文,封装常用的依赖和辅助方法"""

    def __init__(self, plugin: "KnowledgeBasePlugin", event: AstrMessageEvent):
        self.plugin = plugin
        self.event = event
        self.vector_db = plugin.vector_db
        self.config_manager = plugin.config_manager
        self.user_prefs_handler = plugin.user_prefs_handler
        self.text_splitter = plugin.text_splitter
        self.file_parser = plugin.file_parser
        self.search_service = plugin.search_service
        self.rag_service = plugin.rag_service

    async def get_target_collection(
        self, collection_name: Optional[str] = None
    ) -> Optional[str]:
        """
        获取目标知识库名称

        Args:
            collection_name: 指定的知识库名称,如果为 None 则使用用户默认知识库

        Returns:
            str: 目标知识库名称
            None: 如果没有指定且用户也没有默认知识库
        """
        if collection_name:
            return collection_name

        default_collection = self.user_prefs_handler.get_user_default_collection(
            self.event
        )

        # 明确检查 None 和空字符串
        if default_collection is None or default_collection == "":
            return None

        return default_collection

    async def ensure_collection_exists(
        self, collection_name: str, auto_create: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        确保知识库存在

        Args:
            collection_name: 知识库名称
            auto_create: 如果不存在是否自动创建

        Returns:
            tuple[bool, Optional[str]]: (是否存在/创建成功, 错误消息)
                - (True, None): 知识库存在或创建成功
                - (False, error_msg): 知识库不存在且创建失败或未配置自动创建
        """
        exists = await self.vector_db.collection_exists(collection_name)

        if exists:
            logger.debug(f"知识库 '{collection_name}' 已存在")
            return True, None

        # 不存在且不自动创建
        if not auto_create:
            error_msg = f"知识库 '{collection_name}' 不存在"
            logger.warning(error_msg)
            return False, error_msg

        # 尝试自动创建
        try:
            await self.vector_db.create_collection(collection_name)
            logger.info(f"知识库 '{collection_name}' 不存在,已自动创建")
            return True, None
        except Exception as e:
            error_msg = f"自动创建知识库 '{collection_name}' 失败: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def reply(self, message: str) -> AstrMessageEvent:
        """
        创建回复消息

        Args:
            message: 回复内容

        Returns:
            AstrMessageEvent: 可以直接 yield 的事件对象
        """
        return self.event.plain_result(message)

    def format_error(self, operation: str, error: Exception) -> str:
        """
        格式化错误消息

        Args:
            operation: 操作描述(如"添加文本"、"搜索知识库")
            error: 异常对象

        Returns:
            str: 格式化后的错误消息
        """
        return f"{operation}失败: {str(error)}"


class BaseCommandHandler:
    """命令处理器基类"""

    def __init__(self, ctx: CommandContext):
        """
        初始化命令处理器

        Args:
            ctx: 命令上下文
        """
        self.ctx = ctx
        self.plugin = ctx.plugin
        self.event = ctx.event
        self.vector_db = ctx.vector_db
        self.logger = logger

    async def execute(self) -> AsyncGenerator[AstrMessageEvent, None]:
        """
        执行命令 - 子类必须实现

        Yields:
            AstrMessageEvent: 命令执行结果
        """
        raise NotImplementedError("子类必须实现 execute 方法")

    def reply(self, message: str) -> AstrMessageEvent:
        """快捷方法:创建回复消息"""
        return self.ctx.reply(message)

    async def get_target_collection(
        self, collection_name: Optional[str] = None
    ) -> Optional[str]:
        """快捷方法:获取目标知识库"""
        return await self.ctx.get_target_collection(collection_name)

    async def ensure_collection_exists(
        self, collection_name: str, auto_create: bool = False
    ) -> tuple[bool, Optional[str]]:
        """快捷方法:确保知识库存在"""
        return await self.ctx.ensure_collection_exists(collection_name, auto_create)


# ===== 辅助函数 =====


def parse_top_k(top_k_str: Optional[str], default: int = 1, max_value: int = 30) -> int:
    """
    解析 top_k 参数

    Args:
        top_k_str: 字符串形式的 top_k 参数
        default: 默认值
        max_value: 最大值

    Returns:
        int: 解析后的 top_k 值,范围在 [1, max_value]
    """
    if top_k_str is None:
        return default

    # 如果已经是整数
    if isinstance(top_k_str, int):
        return max(1, min(top_k_str, max_value))

    # 尝试转换字符串
    if isinstance(top_k_str, str) and top_k_str.isdigit():
        try:
            top_k = int(top_k_str)
            return max(1, min(top_k, max_value))
        except ValueError:
            logger.warning(
                f"无法将 top_k 参数 '{top_k_str}' 转换为整数,将使用默认值 {default}"
            )
            return default
    else:
        logger.warning(
            f"top_k 参数 '{top_k_str}' (类型: {type(top_k_str)}) 无效,将使用默认值 {default}"
        )
        return default


def validate_non_empty(value: str, field_name: str = "内容") -> tuple[bool, Optional[str]]:
    """
    验证字符串非空

    Args:
        value: 要验证的值
        field_name: 字段名称(用于错误消息)

    Returns:
        tuple[bool, Optional[str]]: (是否有效, 错误消息)
            - (True, None): 验证通过
            - (False, error_msg): 验证失败
    """
    if not value or not value.strip():
        return False, f"{field_name}不能为空"
    return True, None
