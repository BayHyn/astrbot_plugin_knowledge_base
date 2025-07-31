import os
import time
import uuid
import asyncio
import threading
from astrbot.api.star import Context
from .services.document_service import DocumentService
from .services.kb_service import KnowledgeBaseService
from .config.settings import PluginSettings
from .vector_store.base import Document
from quart import request
from astrbot.dashboard.server import Response
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot import logger
from astrbot.core.config.default import VERSION


class KnowledgeBaseWebAPI:
    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        document_service: DocumentService,
        astrbot_context: Context,
        plugin_config: PluginSettings,
    ):
        self.kb_service = kb_service
        self.document_service = document_service
        self.astrbot_context = astrbot_context
        self.plugin_config = plugin_config

        # 从服务中获取依赖
        self.vec_db = self.kb_service.vector_db
        self.user_prefs_handler = self.kb_service.user_prefs_handler
        self.fp = self.document_service.file_parser
        self.text_splitter = self.document_service.text_splitter
        self.tasks = {}
        
        # 文件处理锁，防止并发冲突
        self._file_processing_lock = threading.Lock()
        self._temp_file_counter = 0

        if VERSION < "3.5.13":
            raise RuntimeError("AstrBot 版本过低，无法支持此插件，请升级 AstrBot。")

        # 注册API端点
        self._register_api_endpoints()

    def _register_api_endpoints(self):
        """注册所有API端点，增强容错性"""
        endpoints = [
            ("/alkaid/kb/create_collection", self.create_collection, ["POST"], "创建一个新的知识库集合"),
            ("/alkaid/kb/collections", self.list_collections, ["GET"], "列出所有知识库集合"),
            ("/alkaid/kb/collection/add_file", self.add_documents, ["POST"], "向指定集合添加文档"),
            ("/alkaid/kb/collection/search", self.search_documents, ["GET"], "搜索指定集合中的文档"),
            ("/alkaid/kb/collection/delete", self.delete_collection, ["GET"], "删除指定集合"),
            ("/alkaid/kb/collection/documents", self.list_documents, ["GET"], "获取集合中的文档列表"),
            ("/alkaid/kb/collection/stats", self.get_collection_stats, ["GET"], "获取集合统计信息"),
            ("/alkaid/kb/task_status", self.get_task_status, ["GET"], "获取异步任务的状态"),
        ]
        
        for path, handler, methods, description in endpoints:
            try:
                self.astrbot_context.register_web_api(path, handler, methods, description)
                logger.debug(f"已注册API端点: {path}")
            except Exception as e:
                logger.error(f"注册API端点失败 {path}: {e}")

    def _generate_safe_filename(self, original_filename: str) -> str:
        """生成安全的临时文件名，避免并发冲突"""
        with self._file_processing_lock:
            self._temp_file_counter += 1
            timestamp = int(time.time() * 1000)
            safe_name = f"{timestamp}_{self._temp_file_counter}_{original_filename}"
            return safe_name

    async def _safe_api_call(self, func, *args, **kwargs):
        """安全的API调用包装器"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API调用失败 {func.__name__}: {e}", exc_info=True)
            return Response().error(f"服务内部错误: {str(e)}").__dict__

    async def create_collection(self):
        """
        创建一个新的知识库集合。
        :param collection_name: 集合名称
        :return: 创建结果
        """
        data = await request.get_json()
        collection_name = data.get("collection_name")
        emoji = data.get("emoji", "🙂")
        description = data.get("description", "")
        embedding_provider_id = data.get("embedding_provider_id", None)
        logger.info(f"收到创建知识库请求: {collection_name}")
        if not collection_name:
            return Response().error("缺少集合名称").__dict__
        if await self.vec_db.collection_exists(collection_name):
            return Response().error("集合已存在").__dict__
        if not embedding_provider_id:
            return Response().error("缺少嵌入提供商 ID").__dict__
        try:
            # 添加集合元数据
            metadata = {
                "version": 1,  # metadata 配置版本
                "emoji": emoji,
                "description": description,
                "created_at": int(time.time()),
                "file_id": f"KBDB_{str(uuid.uuid4())}",  # 文件 ID
                "origin": "astrbot-webui",
                "embedding_provider_id": embedding_provider_id,  # AstrBot 嵌入提供商 ID
            }
            collection_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                )
            )
            collection_metadata[collection_name] = metadata
            self.user_prefs_handler.user_collection_preferences[
                "collection_metadata"
            ] = collection_metadata
            await self.user_prefs_handler.save_user_preferences()
            # 兼容性问题，create_collection 方法放在上一步之后执行。
            await self.vec_db.create_collection(collection_name)
            return Response().ok("集合创建成功").__dict__
        except Exception as e:
            return Response().error(f"创建集合失败: {str(e)}").__dict__

    async def list_collections(self):
        """
        列出所有知识库集合。
        :return: 集合列表
        """
        logger.info("收到列出知识库集合请求")
        try:
            collections = await self.vec_db.list_collections()
            result = []
            collections_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                )
            )
            for collection in collections:
                collection_md = collections_metadata.get(collection, {})
                if "embedding_provider_id" in collection_md:
                    p_id = collection_md.get("embedding_provider_id", "")
                    provider = self.astrbot_context.get_provider_by_id(p_id)
                    if provider:
                        collection_md["_embedding_provider_config"] = (
                            provider.provider_config
                        )
                count = await self.vec_db.count_documents(collection)
                result.append(
                    {"collection_name": collection, "count": count, **collection_md}
                )
            return Response().ok(data=result).__dict__
        except Exception as e:
            return Response().error(f"获取集合列表失败: {str(e)}").__dict__

    async def add_documents(self):
        """
        向指定集合添加文档。
        :param collection_name: 集合名称
        :param documents: 文档列表
        :return: 添加结果
        """
        upload_file = (await request.files).get("file")
        collection_name = (await request.form).get("collection_name")
        chunk_size = (await request.form).get("chunk_size", None)
        overlap = (await request.form).get("chunk_overlap", None)

        logger.info(f"收到向知识库 '{collection_name}' 添加文件的请求: {upload_file.filename}")

        if not upload_file or not collection_name:
            return Response().error("缺少知识库名称").__dict__
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        task_id = f"task_{uuid.uuid4()}"
        self.tasks[task_id] = {"status": "pending", "result": None}
        logger.info(f"创建异步任务 {task_id} 用于处理文件 {upload_file.filename}")

        asyncio.create_task(
            self._process_file_asynchronously(
                task_id,
                upload_file,
                collection_name,
                chunk_size,
                overlap,
            )
        )

        return Response().ok(data={"task_id": task_id}, message="文件上传成功，正在后台处理。").__dict__

    async def _process_file_asynchronously(
        self, task_id, upload_file, collection_name, chunk_size_str, overlap_str
    ):
        """异步处理文件，增强容错性和并发安全性"""
        self.tasks[task_id]["status"] = "running"
        temp_path = None
        
        try:
            logger.info(f"[Task {task_id}] 开始处理文件: {upload_file.filename}")
            
            # 参数验证和转换
            try:
                chunk_size = int(chunk_size_str) if chunk_size_str else self.plugin_config.text_chunk_size
                overlap = int(overlap_str) if overlap_str else self.plugin_config.text_chunk_overlap
            except (ValueError, TypeError) as e:
                raise ValueError(f"无效的分块参数: {e}")
            
            # 生成安全的临时文件路径
            temp_dir = os.path.join(get_astrbot_data_path(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            safe_filename = self._generate_safe_filename(upload_file.filename)
            temp_path = os.path.join(temp_dir, safe_filename)
            
            # 保存文件
            try:
                await upload_file.save(temp_path)
                logger.info(f"[Task {task_id}] 文件已保存到临时路径: {temp_path}")
            except Exception as e:
                raise IOError(f"保存文件失败: {e}")

            # 解析文件内容
            try:
                content = await self.fp.parse_file_content(temp_path)
                if not content or not content.strip():
                    raise ValueError("文件内容为空或无法解析")
                logger.info(f"[Task {task_id}] 文件内容解析完成，长度: {len(content)}")
            except Exception as e:
                raise ValueError(f"文件解析失败: {e}")

            # 文本分割
            try:
                chunks = self.text_splitter.split_text(
                    text=content, chunk_size=chunk_size, overlap=overlap
                )
                if not chunks:
                    raise ValueError("文本分割后无有效内容")
                logger.info(f"[Task {task_id}] 文本分割完成，共 {len(chunks)} 个块")
            except Exception as e:
                raise ValueError(f"文本分割失败: {e}")

            # 验证知识库是否仍然存在
            if not await self.vec_db.collection_exists(collection_name):
                raise ValueError(f"目标知识库 '{collection_name}' 不存在")

            # 准备文档
            documents_to_add = [
                Document(
                    text_content=chunk,
                    metadata={
                        "source": upload_file.filename,
                        "user": "astrbot_webui",
                        "upload_time": int(time.time()),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    },
                )
                for i, chunk in enumerate(chunks)
            ]

            # 添加文档到向量数据库
            try:
                doc_ids = await self.vec_db.add_documents(collection_name, documents_to_add)
                if not doc_ids:
                    raise ValueError("向量数据库返回空文档ID列表")
            except Exception as e:
                raise ValueError(f"添加文档到数据库失败: {e}")
            
            success_message = f"成功从文件 '{upload_file.filename}' 添加 {len(doc_ids)} 条知识到 '{collection_name}'。"
            self.tasks[task_id] = {"status": "success", "result": success_message}
            logger.info(f"[Task {task_id}] 任务成功: {success_message}")

        except Exception as e:
            error_message = f"处理文件时发生错误: {str(e)}"
            self.tasks[task_id] = {"status": "failed", "result": error_message}
            logger.error(f"[Task {task_id}] 任务失败: {error_message}", exc_info=True)
        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"[Task {task_id}] 已删除临时文件: {temp_path}")
                except Exception as e:
                    logger.warning(f"[Task {task_id}] 删除临时文件失败: {e}")

    async def search_documents(self):
        """
        搜索指定集合中的文档。
        :param collection_name: 集合名称
        :param query: 查询字符串
        :param top_k: 返回结果数量，默认为5
        :return: 搜索结果
        """
        # 从 URL 参数中获取查询参数
        collection_name = request.args.get("collection_name")
        query = request.args.get("query")
        try:
            top_k = int(request.args.get("top_k", 5))
        except ValueError:
            top_k = 5
        
        logger.info(f"收到在知识库 '{collection_name}' 中搜索的请求: query='{query}', top_k={top_k}")

        # 验证必要参数
        if not collection_name or not query:
            return Response().error("缺少集合名称或查询字符串").__dict__

        # 检查知识库是否存在
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        try:
            # 执行搜索
            results = await self.vec_db.search(collection_name, query, top_k)

            # 格式化结果以便前端展示
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    {
                        "id": result.document.id,
                        "content": result.document.text_content,
                        "metadata": result.document.metadata,
                        "score": result.score,
                    }
                )
            return Response().ok(data=formatted_results).__dict__
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return Response().error(f"搜索失败: {str(e)}").__dict__

    async def delete_collection(self):
        """
        删除指定集合。
        :param collection_name: 集合名称
        """
        # 从 URL 参数中获取查询参数
        collection_name = request.args.get("collection_name")
        logger.info(f"收到删除知识库请求: {collection_name}")

        # 检查知识库是否存在
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        try:
            # 执行删除
            await self.vec_db.delete_collection(collection_name)
            logger.info(f"知识库 '{collection_name}' 删除成功")
            return Response().ok(f"删除 {collection_name} 成功").__dict__
        except Exception as e:
            logger.error(f"删除失败: {str(e)}")
            return Response().error(f"删除失败: {str(e)}").__dict__

    async def get_task_status(self):
        """
        获取异步任务的状态。
        :param task_id: 任务 ID
        :return: 任务状态
        """
        task_id = request.args.get("task_id")
        logger.debug(f"收到获取任务状态请求: {task_id}")
        if not task_id:
            return Response().error("缺少任务 ID").__dict__

        task_info = self.tasks.get(task_id)
        if not task_info:
            return Response().error("任务不存在").__dict__

        logger.debug(f"任务 {task_id} 状态: {task_info}")
        return Response().ok(data=task_info).__dict__

    async def list_documents(self):
        """
        获取指定集合中的文档列表
        """
        collection_name = request.args.get("collection_name")
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 20))
        
        logger.info(f"收到获取文档列表请求: collection={collection_name}, page={page}")
        
        if not collection_name:
            return Response().error("缺少集合名称").__dict__
        
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__
            
        try:
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 获取文档列表（这里需要假设向量数据库支持分页查询）
            # 由于原始接口可能不支持分页，这里提供一个基础实现
            total_count = await self.vec_db.count_documents(collection_name)
            
            # 模拟获取文档列表（实际实现需要根据具体的向量数据库API）
            documents = []
            
            # 基础文档信息
            for i in range(min(page_size, total_count - offset)):
                doc_id = f"doc_{offset + i}"
                documents.append({
                    "id": doc_id,
                    "source": "unknown",  # 需要从元数据中获取
                    "chunk_index": i,
                    "created_at": "unknown",
                    "preview": "文档预览内容..."[:100] + "..."
                })
            
            result = {
                "documents": documents,
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size
            }
            
            return Response().ok(data=result).__dict__
            
        except Exception as e:
            logger.error(f"获取文档列表失败: {str(e)}")
            return Response().error(f"获取文档列表失败: {str(e)}").__dict__

    async def get_collection_stats(self):
        """
        获取集合统计信息
        """
        collection_name = request.args.get("collection_name")
        logger.info(f"收到获取统计信息请求: {collection_name}")
        
        if not collection_name:
            return Response().error("缺少集合名称").__dict__
        
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__
            
        try:
            # 获取基础统计信息
            doc_count = await self.vec_db.count_documents(collection_name)
            
            # 获取集合元数据
            collection_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                ).get(collection_name, {}) if self.user_prefs_handler else {}
            )
            
            # 计算存储大小（估算）
            estimated_size = doc_count * 500  # 每个文档估算500字节
            
            # 统计信息
            stats = {
                "document_count": doc_count,
                "estimated_size_bytes": estimated_size,
                "estimated_size_human": self._format_bytes(estimated_size),
                "created_at": collection_metadata.get("created_at"),
                "last_modified": collection_metadata.get("last_modified", int(time.time())),
                "description": collection_metadata.get("description", ""),
                "emoji": collection_metadata.get("emoji", "📚"),
                "embedding_provider": collection_metadata.get("embedding_provider_id", "unknown")
            }
            
            return Response().ok(data=stats).__dict__
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return Response().error(f"获取统计信息失败: {str(e)}").__dict__

    def _format_bytes(self, bytes_size):
        """格式化字节大小为人类可读格式"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}TB"
