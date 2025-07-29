from typing import List, Optional, AsyncGenerator
from urllib.parse import urlparse
import os

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..vector_store.base import VectorDBBase, Document
from ..utils.file_parser import FileParser
from ..utils.text_splitter import TextSplitterUtil
from ..utils import file_utils
from ..core.constants import ALLOWED_FILE_EXTENSIONS
from .kb_service import KnowledgeBaseService


class DocumentService:
    def __init__(
        self,
        vector_db: VectorDBBase,
        kb_service: KnowledgeBaseService,
        file_parser: FileParser,
        text_splitter: TextSplitterUtil,
        data_path: str,
    ):
        self.vector_db = vector_db
        self.kb_service = kb_service
        self.file_parser = file_parser
        self.text_splitter = text_splitter
        self.data_path = data_path

    async def add_text_to_collection(
        self,
        content: str,
        collection_name: str,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[AstrMessageEvent, None]:
        if not content.strip():
            yield event.plain_result("添加的内容不能为空。")
            return

        await self.kb_service.ensure_collection_exists(collection_name, event)

        chunks = self.text_splitter.split_text(content)
        if not chunks:
            yield event.plain_result("文本分割后无有效内容。")
            return

        documents_to_add = [
            Document(
                text_content=chunk,
                metadata={"source": "direct_text", "user": event.get_sender_name()},
            )
            for chunk in chunks
        ]

        try:
            yield event.plain_result(
                f"正在处理 {len(chunks)} 个文本块并添加到知识库 '{collection_name}'..."
            )
            doc_ids = await self.vector_db.add_documents(
                collection_name, documents_to_add
            )
            if doc_ids:
                yield event.plain_result(
                    f"成功添加 {len(doc_ids)} 条知识到 '{collection_name}'。"
                )
            else:
                yield event.plain_result(
                    f"未能添加任何知识到 '{collection_name}'，请检查日志。"
                )
        except Exception as e:
            logger.error(f"添加文本到知识库 '{collection_name}' 失败: {e}", exc_info=True)
            yield event.plain_result(f"添加知识失败: {e}")

    async def add_file_to_collection(
        self,
        path_or_url: str,
        collection_name: str,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[AstrMessageEvent, None]:
        # This is a simplified version of the original handle_add_file.
        # The full logic will be implemented in subsequent steps.
        yield event.plain_result(f"开始处理文件/URL: {path_or_url}")
        # Placeholder for the full implementation
        pass