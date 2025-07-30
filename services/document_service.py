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
        logger.info(f"收到向知识库 '{collection_name}' 添加文本的请求，发送者: {event.get_sender_name()}")
        if not content.strip():
            logger.warning(f"添加文本内容为空，发送者: {event.get_sender_name()}")
            yield event.plain_result("添加的内容不能为空。")
            return

        logger.info(f"确保知识库 '{collection_name}' 存在")
        await self.kb_service.ensure_collection_exists(collection_name, event)

        logger.info("开始分割文本")
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            logger.warning(f"文本分割后无有效内容，发送者: {event.get_sender_name()}")
            yield event.plain_result("文本分割后无有效内容。")
            return
        logger.info(f"文本分割完成，共 {len(chunks)} 个块")

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
            logger.info(f"开始向知识库 '{collection_name}' 添加 {len(chunks)} 个文档块")
            doc_ids = await self.vector_db.add_documents(
                collection_name, documents_to_add
            )
            if doc_ids:
                logger.info(f"成功添加 {len(doc_ids)} 条知识到 '{collection_name}'")
                yield event.plain_result(
                    f"成功添加 {len(doc_ids)} 条知识到 '{collection_name}'。"
                )
            else:
                logger.warning(f"未能添加任何知识到 '{collection_name}'")
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
        """将文件或URL内容添加到知识库集合中"""
        logger.info(f"收到向知识库 '{collection_name}' 添加文件的请求: {path_or_url}, 发送者: {event.get_sender_name()}")
        try:
            # 验证输入
            if not path_or_url.strip():
                logger.warning(f"文件路径或URL为空，发送者: {event.get_sender_name()}")
                yield event.plain_result("文件路径或URL不能为空。")
                return

            # 确保集合存在
            logger.info(f"确保知识库 '{collection_name}' 存在")
            await self.kb_service.ensure_collection_exists(collection_name, event)

            # 检查是否为URL
            parsed_url = urlparse(path_or_url)
            is_url = bool(parsed_url.scheme and parsed_url.netloc)
            logger.info(f"输入路径类型: {'URL' if is_url else '本地文件'}")

            file_path = None
            temp_file = False

            try:
                if is_url:
                    # 下载文件
                    logger.info(f"开始从URL下载文件: {path_or_url}")
                    yield event.plain_result(f"正在从URL下载文件: {path_or_url}")
                    file_path = await file_utils.download_file(
                        path_or_url, self.data_path
                    )
                    if not file_path:
                        logger.error(f"文件下载失败: {path_or_url}")
                        yield event.plain_result("文件下载失败，请检查URL和文件大小限制。")
                        return
                    temp_file = True
                    logger.info(f"文件下载完成: {file_path}")
                    yield event.plain_result("文件下载完成，开始解析...")
                else:
                    # 本地文件
                    file_path = path_or_url
                    logger.info(f"处理本地文件: {file_path}")
                    if not os.path.exists(file_path):
                        logger.error(f"本地文件不存在: {file_path}")
                        yield event.plain_result(f"文件不存在: {file_path}")
                        return
                    yield event.plain_result("开始解析本地文件...")

                # 检查文件扩展名
                _, extension = os.path.splitext(file_path)
                extension = extension.lower()
                if extension not in ALLOWED_FILE_EXTENSIONS:
                    logger.warning(f"不支持的文件类型: {extension}")
                    yield event.plain_result(
                        f"不支持的文件类型: {extension}。支持的类型: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
                    )
                    return

                # 解析文件内容
                logger.info(f"开始解析文件内容: {file_path}")
                content = await self.file_parser.parse_file_content(file_path)
                if not content or not content.strip():
                    logger.warning(f"文件内容为空或无法解析: {file_path}")
                    yield event.plain_result("文件内容为空或无法解析。")
                    return
                logger.info(f"文件内容解析完成，长度: {len(content)}")

                # 文本分割
                logger.info("开始分割文本")
                chunks = self.text_splitter.split_text(content)
                if not chunks:
                    logger.warning("文本分割后无有效内容")
                    yield event.plain_result("文本分割后无有效内容。")
                    return
                logger.info(f"文本分割完成，共 {len(chunks)} 个块")

                # 准备文档
                source_name = os.path.basename(file_path) if not is_url else path_or_url
                documents_to_add = [
                    Document(
                        text_content=chunk,
                        metadata={
                            "source": source_name,
                            "user": event.get_sender_name(),
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        },
                    )
                    for i, chunk in enumerate(chunks)
                ]

                # 添加到知识库
                yield event.plain_result(
                    f"正在处理 {len(chunks)} 个文本块并添加到知识库 '{collection_name}'..."
                )
                logger.info(f"开始向知识库 '{collection_name}' 添加 {len(chunks)} 个文档块")
                doc_ids = await self.vector_db.add_documents(
                    collection_name, documents_to_add
                )

                if doc_ids:
                    logger.info(f"成功添加 {len(doc_ids)} 条知识到 '{collection_name}'")
                    yield event.plain_result(
                        f"成功添加 {len(doc_ids)} 条知识到 '{collection_name}'。\n"
                        f"来源: {source_name}"
                    )
                else:
                    logger.warning(f"未能添加任何知识到 '{collection_name}'")
                    yield event.plain_result(
                        f"未能添加任何知识到 '{collection_name}'，请检查日志。"
                    )

            finally:
                # 清理临时文件
                if temp_file and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"已删除临时文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {e}")

        except Exception as e:
            logger.error(
                f"添加文件到知识库 '{collection_name}' 失败: {e}",
                exc_info=True
            )
            yield event.plain_result(f"添加文件失败: {str(e)}")