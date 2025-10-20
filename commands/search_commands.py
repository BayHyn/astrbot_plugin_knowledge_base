# astrbot_plugin_knowledge_base/command_handlers/search_commands.py
from typing import Optional, TYPE_CHECKING, AsyncGenerator
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from .base import CommandContext, BaseCommandHandler, parse_top_k, validate_non_empty

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin


class SearchCommandHandler(BaseCommandHandler):
    """搜索命令处理器"""

    def __init__(
        self,
        ctx: CommandContext,
        query: str,
        collection_name: Optional[str] = None,
        top_k_str: Optional[str] = None,
    ):
        super().__init__(ctx)
        self.query = query
        self.collection_name = collection_name
        self.top_k = parse_top_k(top_k_str, default=1, max_value=30)

    async def execute(self) -> AsyncGenerator[AstrMessageEvent, None]:
        """执行搜索命令"""
        # 1. 验证查询内容
        is_valid, error_msg = validate_non_empty(self.query, "查询内容")
        if not is_valid:
            yield self.reply(error_msg)
            return

        # 2. 获取目标知识库
        target_collection = await self.get_target_collection(self.collection_name)
        if not target_collection:
            yield self.reply("请先设置默认知识库或指定要搜索的知识库。")
            return

        # 3. 检查知识库是否存在
        exists, error_msg = await self.ensure_collection_exists(target_collection)
        if not exists:
            yield self.reply(error_msg)
            return

        # 4. 执行搜索
        logger.info(
            f"搜索知识库 '{target_collection}',查询: '{self.query[:30]}...', "
            f"top_k: {self.top_k}"
        )

        try:
            yield self.reply(
                f"正在知识库 '{target_collection}' 中搜索 "
                f"'{self.query[:30]}...' (最多{self.top_k}条)..."
            )

            # 使用 SearchService
            search_results = await self.ctx.search_service.search(
                collection_name=target_collection,
                query=self.query,
                top_k=self.top_k,
            )

            if not search_results:
                yield self.reply(
                    f"在知识库 '{target_collection}' 中未找到与 "
                    f"'{self.query[:30]}...' 相关的内容。"
                )
                return

            # 5. 格式化结果
            response_message = (
                f"知识库 '{target_collection}' 中关于 '{self.query[:30]}...' "
                f"的搜索结果 (相关度从高到低):\n"
            )

            for i, (doc, score) in enumerate(search_results):
                source_info = (
                    f" (来源: {doc.metadata.get('source', '未知')})"
                    if doc.metadata.get("source")
                    else ""
                )
                response_message += f"\n{i + 1}. [相关度: {score:.2f}]{source_info}\n"
                content_preview = (
                    doc.text_content[:200] + "..."
                    if len(doc.text_content) > 200
                    else doc.text_content
                )
                response_message += f"   内容: {content_preview}\n"

            # 如果结果太长,尝试转为图片
            if len(response_message) > 1500:
                yield self.reply("搜索结果较长,将尝试转为图片发送。")
                try:
                    img_url = await self.plugin.text_to_image(response_message)
                    yield self.event.image_result(img_url)
                except Exception as img_error:
                    logger.warning(f"转换图片失败: {img_error},将以文本形式发送")
                    yield self.reply(response_message)
            else:
                yield self.reply(response_message)

            logger.info(
                f"搜索完成,返回 {len(search_results)} 条结果 "
                f"(知识库: '{target_collection}', 查询: '{self.query[:30]}...')"
            )

        except Exception as e:
            error_msg = self.ctx.format_error("搜索知识库", e)
            logger.error(
                f"搜索知识库 '{target_collection}' 失败: {e}", exc_info=True
            )
            yield self.reply(error_msg)


# ===== 兼容层函数 =====


async def handle_search(
    plugin: "KnowledgeBasePlugin",
    event: AstrMessageEvent,
    query: str,
    collection_name: Optional[str] = None,
    top_k_str: Optional[str] = None,
) -> AsyncGenerator[AstrMessageEvent, None]:
    """
    搜索命令处理函数 (兼容层)

    这是一个兼容层函数,保持原有的函数签名以避免破坏现有代码。
    实际业务逻辑已经迁移到 SearchCommandHandler 类中。
    """
    ctx = CommandContext(plugin, event)
    handler = SearchCommandHandler(ctx, query, collection_name, top_k_str)
    async for result in handler.execute():
        yield result
