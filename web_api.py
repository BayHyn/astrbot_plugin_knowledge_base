import os
import time
import uuid
import asyncio
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

        if VERSION < "3.5.13":
            raise RuntimeError("AstrBot 版本过低，无法支持此插件，请升级 AstrBot。")

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
            "/alkaid/kb/task_status",
            self.get_task_status,
            ["GET"],
            "获取异步任务的状态",
        )

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
        self.tasks[task_id]["status"] = "running"
        path = None
        try:
            logger.info(f"[Task {task_id}] 开始处理文件: {upload_file.filename}")
            chunk_size = int(chunk_size_str) if chunk_size_str else None
            overlap = int(overlap_str) if overlap_str else None
            
            temp_dir = os.path.join(get_astrbot_data_path(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            path = os.path.join(temp_dir, upload_file.filename)
            
            await upload_file.save(path)
            logger.info(f"[Task {task_id}] 文件已保存到临时路径: {path}")

            content = await self.fp.parse_file_content(path)
            if not content:
                raise ValueError("文件内容为空或不支持的格式")
            logger.info(f"[Task {task_id}] 文件内容解析完成，长度: {len(content)}")

            chunks = self.text_splitter.split_text(
                text=content, chunk_size=chunk_size, overlap=overlap
            )
            if not chunks:
                raise Exception("文本分割后无有效内容 (chunk 内容为空)")
            logger.info(f"[Task {task_id}] 文本分割完成，共 {len(chunks)} 个块")

            documents_to_add = [
                Document(
                    text_content=chunk,
                    metadata={"source": upload_file.filename, "user": "astrbot_webui"},
                )
                for chunk in chunks
            ]

            doc_ids = await self.vec_db.add_documents(collection_name, documents_to_add)
            if not doc_ids:
                raise Exception("添加文档到数据库失败，返回的文档 ID 为空")
            
            message = f"成功从文件 '{upload_file.filename}' 添加 {len(doc_ids)} 条知识到 '{collection_name}'。"
            self.tasks[task_id] = {"status": "success", "result": message}
            logger.info(f"[Task {task_id}] 任务成功: {message}")

        except Exception as e:
            error_message = f"处理文件时发生错误: {str(e)}"
            self.tasks[task_id] = {"status": "failed", "result": error_message}
            logger.error(f"[Task {task_id}] 任务失败: {error_message}", exc_info=True)
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"[Task {task_id}] 已删除临时文件: {path}")
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
