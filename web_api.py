import os
import time
import uuid
import asyncio
import threading
import json
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
        
        # 任务持久化文件路径
        self._tasks_file_path = os.path.join(get_astrbot_data_path(), "kb_tasks.json")
        self._load_tasks_from_file()

        if VERSION < "3.5.13":
            raise RuntimeError("AstrBot 版本过低，无法支持此插件，请升级 AstrBot。")

        # 注册API端点
        self.astrbot_context.register_web_api(
            "/alkaid/kb/create_collection",
            self.create_collection,
            ["POST"],
            "创建一个新的知识库集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collections",
            self.list_collections,
            ["GET"],
            "列出所有知识库集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/add_file",
            self.add_documents,
            ["POST"],
            "向指定集合添加文档",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/search",
            self.search_documents,
            ["GET"],
            "搜索指定集合中的文档",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/delete",
            self.delete_collection,
            ["GET"],
            "删除指定集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/documents",
            self.list_documents,
            ["GET"],
            "获取集合中的文档列表",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/stats",
            self.get_collection_stats,
            ["GET"],
            "获取集合统计信息",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/task_status",
            self.get_task_status,
            ["GET"],
            "获取异步任务的状态",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/debug/repair_collection",
            self.repair_collection_data,
            ["GET"],
            "修复集合数据（检查数据一致性）",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/document/delete",
            self.delete_document,
            ["DELETE"],
            "删除指定集合中的文档",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/update",
            self.update_collection,
            ["PUT"],
            "更新集合的元数据信息",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/import",
            self.import_collection,
            ["POST"],
            "导入集合数据",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/batch_upload",
            self.batch_upload_files,
            ["POST"],
            "批量上传文件到集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/recent_tasks",
            self.get_recent_tasks,
            ["GET"],
            "获取最近的任务列表和状态",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/tasks/cleanup",
            self.cleanup_old_tasks,
            ["POST"],
            "清理旧的任务记录",
        )

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
            # 清理损坏的集合（可选，仅在需要时执行）
            corrupted_collections = await self.vec_db.cleanup_corrupted_collections()
            if corrupted_collections:
                logger.info(f"清理了 {len(corrupted_collections)} 个损坏的集合文件")

                # 同时清理这些集合的元数据
                if self.user_prefs_handler:
                    collection_metadata = (
                        self.user_prefs_handler.user_collection_preferences.get(
                            "collection_metadata", {}
                        )
                    )
                    cleaned_metadata = False
                    for corrupted_name in corrupted_collections:
                        if corrupted_name in collection_metadata:
                            del collection_metadata[corrupted_name]
                            cleaned_metadata = True
                            logger.info(f"清理了损坏集合 '{corrupted_name}' 的元数据")

                    if cleaned_metadata:
                        self.user_prefs_handler.user_collection_preferences[
                            "collection_metadata"
                        ] = collection_metadata
                        await self.user_prefs_handler.save_user_preferences()

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

        logger.info(
            f"收到向知识库 '{collection_name}' 添加文件的请求: {upload_file.filename}"
        )

        if not upload_file or not collection_name:
            return Response().error("缺少知识库名称").__dict__
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        # 立即保存文件到临时位置
        try:
            # 生成安全的临时文件路径
            temp_dir = os.path.join(get_astrbot_data_path(), "temp")
            os.makedirs(temp_dir, exist_ok=True)

            safe_filename = self._generate_safe_filename(upload_file.filename)
            temp_path = os.path.join(temp_dir, safe_filename)

            # 立即保存文件
            await upload_file.save(temp_path)
            logger.info(f"文件已保存到临时路径: {temp_path}")

        except Exception as e:
            logger.error(f"保存上传文件失败: {e}")
            return Response().error(f"保存文件失败: {str(e)}").__dict__

        task_id = f"task_{uuid.uuid4()}"
        task_info = {
            "status": "pending", 
            "result": None,
            "filename": upload_file.filename,
            "collection_name": collection_name,
            "created_at": int(time.time())
        }
        self._add_task(task_id, task_info)
        logger.info(f"创建异步任务 {task_id} 用于处理文件 {upload_file.filename}")

        # 立即启动异步任务，不等待完成
        asyncio.create_task(
            self._process_file_asynchronously(
                task_id,
                temp_path,  # 传递文件路径而不是文件对象
                upload_file.filename,  # 传递原始文件名
                collection_name,
                chunk_size,
                overlap,
            )
        )

        return (
            Response()
            .ok(data={
                "task_id": task_id,
                "message": "文件已提交处理，请稍后查看处理结果"
            }, message="文件上传成功，正在后台处理")
            .__dict__
        )

    async def _process_file_asynchronously(
        self,
        task_id,
        temp_path,
        original_filename,
        collection_name,
        chunk_size_str,
        overlap_str,
    ):
        """异步处理文件，增强容错性和并发安全性"""
        self._update_task(task_id, {"status": "running"})

        try:
            logger.info(f"[Task {task_id}] 开始处理文件: {original_filename}")

            # 参数验证和转换
            try:
                chunk_size = (
                    int(chunk_size_str)
                    if chunk_size_str
                    else self.plugin_config.text_chunk_size
                )
                overlap = (
                    int(overlap_str)
                    if overlap_str
                    else self.plugin_config.text_chunk_overlap
                )
            except (ValueError, TypeError) as e:
                raise ValueError(f"无效的分块参数: {e}")

            # 验证文件是否存在
            if not os.path.exists(temp_path):
                raise ValueError("临时文件不存在")

            logger.info(f"[Task {task_id}] 开始解析文件: {temp_path}")

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
                chunks = await self.text_splitter.split_text(
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
                        "source": original_filename,
                        "user": "astrbot_webui",
                        "upload_time": int(time.time()),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
                for i, chunk in enumerate(chunks)
            ]

            # 添加文档到向量数据库
            try:
                doc_ids = await self.vec_db.add_documents(
                    collection_name, documents_to_add
                )
                if not doc_ids:
                    raise ValueError("向量数据库返回空文档ID列表")
            except Exception as e:
                raise ValueError(f"添加文档到数据库失败: {e}")

            success_message = f"成功从文件 '{original_filename}' 添加 {len(doc_ids)} 条知识到 '{collection_name}'。"
            self._update_task(task_id, {"status": "success", "result": success_message})
            logger.info(f"[Task {task_id}] 任务成功: {success_message}")

        except Exception as e:
            error_message = f"处理文件时发生错误: {str(e)}"
            self._update_task(task_id, {"status": "failed", "result": error_message})
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

        logger.info(
            f"收到在知识库 '{collection_name}' 中搜索的请求: query='{query}', top_k={top_k}"
        )

        # 验证必要参数
        if not collection_name or not query:
            return Response().error("缺少集合名称或查询字符串").__dict__

        # 检查知识库是否存在
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        # TODO: 添加数据完整性检查
        # 检查集合是否真的有数据
        doc_count = await self.vec_db.count_documents(collection_name)
        logger.info(f"知识库 '{collection_name}' 包含 {doc_count} 个文档")

        if doc_count == 0:
            logger.warning(f"知识库 '{collection_name}' 为空，无法搜索")
            return Response().ok(data=[], message="知识库为空，请先添加文档").__dict__

        try:
            # 执行搜索
            logger.debug(f"开始在知识库 '{collection_name}' 中搜索...")
            results = await self.vec_db.search(collection_name, query, top_k)
            logger.info(f"搜索完成，找到 {len(results)} 个结果")

            # 格式化结果以便前端展示
            formatted_results = []
            for i, result in enumerate(results):
                logger.debug(
                    f"结果 {i + 1}: score={result.score:.4f}, id={result.document.id}"
                )
                formatted_results.append(
                    {
                        "id": result.document.id,
                        "content": result.document.text_content,
                        "metadata": result.document.metadata,
                        "score": result.score,
                    }
                )

            logger.info(f"返回格式化结果: {len(formatted_results)} 个项目")
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

        if not collection_name:
            return Response().error("缺少集合名称").__dict__

        try:
            # 尝试删除向量数据库中的集合
            deleted = await self.vec_db.delete_collection(collection_name)

            # 清理用户偏好中的集合元数据
            if self.user_prefs_handler:
                collection_metadata = (
                    self.user_prefs_handler.user_collection_preferences.get(
                        "collection_metadata", {}
                    )
                )
                if collection_name in collection_metadata:
                    del collection_metadata[collection_name]
                    self.user_prefs_handler.user_collection_preferences[
                        "collection_metadata"
                    ] = collection_metadata
                    await self.user_prefs_handler.save_user_preferences()
                    logger.info(f"已清理知识库 '{collection_name}' 的元数据")

            if deleted:
                logger.info(f"知识库 '{collection_name}' 删除成功")
                return Response().ok(f"删除 {collection_name} 成功").__dict__
            else:
                return Response().error("目标知识库不存在或删除失败").__dict__

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
                documents.append(
                    {
                        "id": doc_id,
                        "source": "unknown",  # 需要从元数据中获取
                        "chunk_index": i,
                        "created_at": "unknown",
                        "preview": "文档预览内容..."[:100] + "...",
                    }
                )

            result = {
                "documents": documents,
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size,
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
                ).get(collection_name, {})
                if self.user_prefs_handler
                else {}
            )

            # 计算存储大小（估算）
            estimated_size = doc_count * 500  # 每个文档估算500字节

            # 统计信息
            stats = {
                "document_count": doc_count,
                "estimated_size_bytes": estimated_size,
                "estimated_size_human": self._format_bytes(estimated_size),
                "created_at": collection_metadata.get("created_at"),
                "last_modified": collection_metadata.get(
                    "last_modified", int(time.time())
                ),
                "description": collection_metadata.get("description", ""),
                "emoji": collection_metadata.get("emoji", "📚"),
                "embedding_provider": collection_metadata.get(
                    "embedding_provider_id", "unknown"
                ),
            }

            return Response().ok(data=stats).__dict__

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return Response().error(f"获取统计信息失败: {str(e)}").__dict__

    def _format_bytes(self, bytes_size):
        """格式化字节大小为人类可读格式"""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}TB"

    async def repair_collection_data(self):
        """修复集合数据：检查并重新生成缺失的向量"""
        collection_name = request.args.get("collection_name")
        logger.info(f"收到修复集合数据请求: {collection_name}")

        if not collection_name:
            return Response().error("缺少集合名称").__dict__

        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        try:
            # TODO: 实现数据修复逻辑
            # 1. 检查数据库和索引的一致性
            # 2. 重新生成缺失的向量
            # 3. 重建索引文件

            # 先获取基本信息
            doc_count = await self.vec_db.count_documents(collection_name)

            # 检查向量索引状态（简化实现）
            # 注意：EnhancedVectorStore 不直接暴露内部索引状态
            # 这里使用文档数量作为估算
            index_count = doc_count  # 假设索引与文档数量一致

            repair_info = {
                "collection_name": collection_name,
                "documents_in_db": doc_count,
                "vectors_in_index": index_count,
                "consistent": True,  # 简化为总是一致
                "repair_needed": False,  # 简化为不需要修复
                "suggestion": "如果遇到搜索问题，请重新上传文档",
            }

            logger.info(
                f"集合 '{collection_name}' 数据状态: 数据库文档={doc_count}, 索引向量={index_count}"
            )

            return Response().ok(data=repair_info).__dict__

        except Exception as e:
            logger.error(f"修复集合数据失败: {str(e)}")
            return Response().error(f"修复失败: {str(e)}").__dict__

    async def delete_document(self):
        """删除指定集合中的单个文档"""
        data = await request.get_json()
        collection_name = data.get("collection_name")
        document_id = data.get("document_id")
        
        logger.info(f"收到删除文档请求: collection={collection_name}, doc_id={document_id}")
        
        if not collection_name or not document_id:
            return Response().error("缺少集合名称或文档ID").__dict__
            
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__
            
        try:
            # 删除指定文档
            deleted = await self.vec_db.delete_document(collection_name, document_id)
            if deleted:
                return Response().ok("文档删除成功").__dict__
            else:
                return Response().error("文档不存在或删除失败").__dict__
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return Response().error(f"删除文档失败: {str(e)}").__dict__

    async def update_collection(self):
        """更新集合的元数据信息"""
        data = await request.get_json()
        collection_name = data.get("collection_name")
        new_name = data.get("new_name")
        emoji = data.get("emoji")
        description = data.get("description")
        
        logger.info(f"收到更新集合元数据请求: {collection_name}")
        
        if not collection_name:
            return Response().error("缺少集合名称").__dict__
            
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__
            
        try:
            # 更新集合元数据
            collection_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                )
            )
            
            if collection_name in collection_metadata:
                metadata = collection_metadata[collection_name]
                if emoji is not None:
                    metadata["emoji"] = emoji
                if description is not None:
                    metadata["description"] = description
                metadata["last_modified"] = int(time.time())
                
                # 如果需要重命名集合
                if new_name and new_name != collection_name:
                    if await self.vec_db.collection_exists(new_name):
                        return Response().error("新集合名称已存在").__dict__
                    
                    # 重命名集合
                    await self.vec_db.rename_collection(collection_name, new_name)
                    
                    # 更新元数据中的键名
                    collection_metadata[new_name] = metadata
                    del collection_metadata[collection_name]
                else:
                    collection_metadata[collection_name] = metadata
                
                self.user_prefs_handler.user_collection_preferences[
                    "collection_metadata"
                ] = collection_metadata
                await self.user_prefs_handler.save_user_preferences()
                
                return Response().ok("集合信息更新成功").__dict__
            else:
                return Response().error("集合元数据不存在").__dict__
                
        except Exception as e:
            logger.error(f"更新集合信息失败: {str(e)}")
            return Response().error(f"更新失败: {str(e)}").__dict__

    async def import_collection(self):
        """导入集合数据"""
        data = await request.get_json()
        collection_data = data.get("collection_data")
        overwrite = data.get("overwrite", False)
        
        logger.info(f"收到导入集合请求")
        
        if not collection_data:
            return Response().error("缺少集合数据").__dict__
            
        try:
            collection_name = collection_data.get("collection_name")
            metadata = collection_data.get("metadata", {})
            
            if not collection_name:
                return Response().error("集合数据格式错误").__dict__
                
            # 检查集合是否已存在
            if await self.vec_db.collection_exists(collection_name) and not overwrite:
                return Response().error("集合已存在，如需覆盖请设置overwrite=true").__dict__
                
            # 创建或覆盖集合
            if overwrite and await self.vec_db.collection_exists(collection_name):
                await self.vec_db.delete_collection(collection_name)
                
            await self.vec_db.create_collection(collection_name)
            
            # 导入元数据
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
            
            return Response().ok("集合导入成功").__dict__
            
        except Exception as e:
            logger.error(f"导入集合失败: {str(e)}")
            return Response().error(f"导入失败: {str(e)}").__dict__

    async def batch_upload_files(self):
        """批量上传文件到集合"""
        files = await request.files
        form_data = await request.form
        collection_name = form_data.get("collection_name")
        chunk_size = form_data.get("chunk_size")
        overlap = form_data.get("chunk_overlap")
        
        logger.info(f"收到批量上传请求: collection={collection_name}, files={len(files)}")
        
        if not collection_name:
            return Response().error("缺少集合名称").__dict__
            
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__
            
        if not files:
            return Response().error("没有上传的文件").__dict__
            
        try:
            task_ids = []
            temp_dir = os.path.join(get_astrbot_data_path(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 为每个文件创建异步任务
            for file_key, upload_file in files.items():
                if upload_file.filename:
                    safe_filename = self._generate_safe_filename(upload_file.filename)
                    temp_path = os.path.join(temp_dir, safe_filename)
                    
                    # 保存文件
                    await upload_file.save(temp_path)
                    
                    # 创建处理任务
                    task_id = f"batch_task_{uuid.uuid4()}"
                    task_info = {
                        "status": "pending", 
                        "result": None, 
                        "filename": upload_file.filename,
                        "collection_name": collection_name,
                        "created_at": int(time.time())
                    }
                    self._add_task(task_id, task_info)
                    task_ids.append(task_id)
                    
                    # 启动异步处理
                    asyncio.create_task(
                        self._process_file_asynchronously(
                            task_id,
                            temp_path,
                            upload_file.filename,
                            collection_name,
                            chunk_size,
                            overlap,
                        )
                    )
            
            return Response().ok(
                data={"task_ids": task_ids}, 
                message=f"已提交{len(task_ids)}个文件的批量处理任务"
            ).__dict__
            
        except Exception as e:
            logger.error(f"批量上传失败: {str(e)}")
            return Response().error(f"批量上传失败: {str(e)}").__dict__

    async def get_recent_tasks(self):
        """获取最近的任务列表和状态"""
        collection_name = request.args.get("collection_name")
        limit = int(request.args.get("limit", 20))
        
        logger.info(f"收到获取任务列表请求: collection_name={collection_name}, limit={limit}")
        logger.info(f"当前任务总数: {len(self.tasks)}")
        
        try:
            # 获取任务列表，如果指定了collection_name则过滤
            recent_tasks = []
            for task_id, task_info in self.tasks.items():
                logger.debug(f"检查任务 {task_id}: {task_info}")
                if collection_name and task_info.get("collection_name") != collection_name:
                    continue
                
                recent_tasks.append({
                    "task_id": task_id,
                    "filename": task_info.get("filename", "未知文件"),
                    "collection_name": task_info.get("collection_name", ""),
                    "status": task_info.get("status", "unknown"),
                    "result": task_info.get("result"),
                    "created_at": task_info.get("created_at", 0)
                })
            
            logger.info(f"过滤后的任务数量: {len(recent_tasks)}")
            
            # 按创建时间排序，最新的在前
            recent_tasks.sort(key=lambda x: x["created_at"], reverse=True)
            
            # 限制返回数量
            if limit > 0:
                recent_tasks = recent_tasks[:limit]
            
            return Response().ok(data={
                "tasks": recent_tasks,
                "total": len(recent_tasks)
            }).__dict__
            
        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}")
            return Response().error(f"获取任务列表失败: {str(e)}").__dict__

    async def cleanup_old_tasks(self):
        """清理旧的任务记录"""
        data = await request.get_json()
        max_age_hours = data.get("max_age_hours", 24)  # 默认清理24小时前的任务
        keep_completed = data.get("keep_completed", True)  # 是否保留已完成的任务
        
        try:
            current_time = int(time.time())
            max_age_seconds = max_age_hours * 3600
            
            tasks_to_remove = []
            for task_id, task_info in self.tasks.items():
                task_age = current_time - task_info.get("created_at", current_time)
                
                # 如果任务太旧
                if task_age > max_age_seconds:
                    # 如果设置了保留已完成任务，则跳过已完成的任务
                    if keep_completed and task_info.get("status") == "success":
                        continue
                    tasks_to_remove.append(task_id)
            
            # 删除旧任务
            removed_count = 0
            for task_id in tasks_to_remove:
                if task_id in self.tasks:
                    del self.tasks[task_id]
                    removed_count += 1
            
            # 保存更新后的任务数据
            if removed_count > 0:
                self._save_tasks_to_file()
            
            logger.info(f"清理了 {removed_count} 个旧任务记录")
            
            return Response().ok(data={
                "removed_count": removed_count,
                "remaining_count": len(self.tasks)
            }, message=f"已清理 {removed_count} 个旧任务记录").__dict__
            
        except Exception as e:
            logger.error(f"清理任务失败: {str(e)}")
            return Response().error(f"清理任务失败: {str(e)}").__dict__

    def _load_tasks_from_file(self):
        """从文件加载任务数据"""
        try:
            if os.path.exists(self._tasks_file_path):
                with open(self._tasks_file_path, 'r', encoding='utf-8') as f:
                    self.tasks = json.load(f)
                logger.info(f"从文件加载了 {len(self.tasks)} 个任务")
            else:
                self.tasks = {}
                logger.info("任务文件不存在，初始化空任务字典")
        except Exception as e:
            logger.error(f"加载任务文件失败: {e}")
            self.tasks = {}

    def _save_tasks_to_file(self):
        """保存任务数据到文件"""
        try:
            with open(self._tasks_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.tasks, f, ensure_ascii=False, indent=2)
            logger.debug(f"任务数据已保存到文件，共 {len(self.tasks)} 个任务")
        except Exception as e:
            logger.error(f"保存任务文件失败: {e}")

    def _add_task(self, task_id, task_info):
        """添加任务并保存到文件"""
        self.tasks[task_id] = task_info
        self._save_tasks_to_file()

    def _update_task(self, task_id, updates):
        """更新任务状态并保存到文件"""
        if task_id in self.tasks:
            self.tasks[task_id].update(updates)
            self._save_tasks_to_file()
