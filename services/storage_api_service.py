"""
知识库存储底层API服务
提供对外开放的知识库存储功能接口，支持增删改查、分块控制、关键词提取等功能
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..vector_store.base import VectorDBBase, Document, DocumentMetadata, Filter, FilterCondition, SearchResult
from ..services.kb_service import KnowledgeBaseService
from ..utils.text_splitter import TextSplitterUtil
from ..utils.file_parser import FileParser
from ..config.settings import PluginSettings


@dataclass
class StorageOptions:
    """存储选项配置"""
    enable_chunking: bool = True  # 是否启用文本分块
    chunk_size: int = 1000  # 分块大小
    chunk_overlap: int = 200  # 分块重叠
    extract_keywords: bool = False  # 是否提取关键词
    auto_create_collection: bool = True  # 是否自动创建知识库
    custom_metadata: Optional[Dict[str, Any]] = None  # 自定义元数据


@dataclass
class StorageResult:
    """存储操作结果"""
    success: bool
    message: str
    doc_ids: Optional[List[str]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryOptions:
    """查询选项配置"""
    top_k: int = 5  # 返回结果数量
    similarity_threshold: float = 0.0  # 相似度阈值
    enable_rerank: bool = False  # 是否启用重排序
    filters: Optional[Filter] = None  # 元数据过滤器
    include_metadata: bool = True  # 是否包含元数据


@dataclass
class QueryResult:
    """查询结果"""
    success: bool
    message: str
    results: Optional[List[SearchResult]] = None
    total_count: int = 0
    error: Optional[str] = None


class StorageAPIService:
    """知识库存储API服务"""
    
    def __init__(
        self,
        vector_db: VectorDBBase,
        kb_service: KnowledgeBaseService,
        text_splitter: TextSplitterUtil,
        file_parser: Optional[FileParser] = None,
        settings: Optional[PluginSettings] = None,
    ):
        self.vector_db = vector_db
        self.kb_service = kb_service
        self.text_splitter = text_splitter
        self.file_parser = file_parser
        self.settings = settings or PluginSettings()
        
    async def add_text(
        self,
        collection_name: str,
        text_content: str,
        options: Optional[StorageOptions] = None,
        source_info: Optional[Dict[str, Any]] = None,
    ) -> StorageResult:
        """
        添加文本到知识库
        
        Args:
            collection_name: 知识库名称
            text_content: 文本内容
            options: 存储选项
            source_info: 来源信息 (如用户信息、文件名等)
            
        Returns:
            StorageResult: 存储结果
        """
        try:
            if not text_content.strip():
                return StorageResult(
                    success=False,
                    message="文本内容不能为空",
                    error="Empty text content"
                )
            
            options = options or StorageOptions()
            
            # 确保知识库存在
            if options.auto_create_collection:
                if not await self.vector_db.collection_exists(collection_name):
                    await self.vector_db.create_collection(collection_name)
                    logger.info(f"自动创建知识库: {collection_name}")
            
            # 文本分块处理
            if options.enable_chunking:
                # 使用自定义分块参数
                if options.chunk_size != self.text_splitter.chunk_size:
                    self.text_splitter.chunk_size = options.chunk_size
                if options.chunk_overlap != self.text_splitter.chunk_overlap:
                    self.text_splitter.chunk_overlap = options.chunk_overlap
                
                chunks = await self.text_splitter.split_text(text_content)
            else:
                chunks = [text_content]
            
            if not chunks:
                return StorageResult(
                    success=False,
                    message="文本分块后无有效内容",
                    error="No valid chunks after splitting"
                )
            
            # 准备文档元数据
            base_metadata = {
                "created_at": time.time(),
                "chunk_count": len(chunks),
                "enable_chunking": options.enable_chunking,
                "chunk_size": options.chunk_size,
                "chunk_overlap": options.chunk_overlap,
            }
            
            # 合并来源信息
            if source_info:
                base_metadata.update(source_info)
            
            # 合并自定义元数据
            if options.custom_metadata:
                base_metadata.update(options.custom_metadata)
            
            # 创建文档列表
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                })
                
                # 关键词提取 (如果启用)
                if options.extract_keywords:
                    keywords = await self._extract_keywords(chunk)
                    chunk_metadata["keywords"] = keywords
                
                doc = Document(
                    text_content=chunk,
                    metadata=DocumentMetadata(
                        source=source_info.get("source", "api_call") if source_info else "api_call",
                        created_at=time.time(),
                        custom_fields=chunk_metadata
                    )
                )
                documents.append(doc)
            
            # 添加到向量数据库
            doc_ids = await self.vector_db.add_documents(collection_name, documents)
            
            return StorageResult(
                success=True,
                message=f"成功添加 {len(doc_ids)} 个文档块到知识库 '{collection_name}'",
                doc_ids=doc_ids,
                metadata={
                    "chunk_count": len(chunks),
                    "collection_name": collection_name,
                    "processing_options": options.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"添加文本到知识库失败: {e}", exc_info=True)
            return StorageResult(
                success=False,
                message=f"添加文本失败: {str(e)}",
                error=str(e)
            )
    
    async def search_knowledge(
        self,
        collection_name: str,
        query_text: str,
        options: Optional[QueryOptions] = None,
    ) -> QueryResult:
        """
        在知识库中搜索相关内容
        
        Args:
            collection_name: 知识库名称
            query_text: 查询文本
            options: 查询选项
            
        Returns:
            QueryResult: 查询结果
        """
        try:
            if not query_text.strip():
                return QueryResult(
                    success=False,
                    message="查询文本不能为空",
                    error="Empty query text"
                )
            
            options = options or QueryOptions()
            
            # 检查知识库是否存在
            if not await self.vector_db.collection_exists(collection_name):
                return QueryResult(
                    success=False,
                    message=f"知识库 '{collection_name}' 不存在",
                    error="Collection not found"
                )
            
            # 执行搜索
            search_results = await self.vector_db.search(
                collection_name=collection_name,
                query_text=query_text,
                top_k=options.top_k,
                filters=options.filters
            )
            
            # 应用相似度阈值过滤
            if options.similarity_threshold > 0:
                search_results = [
                    result for result in search_results 
                    if result.score >= options.similarity_threshold
                ]
            
            return QueryResult(
                success=True,
                message=f"在知识库 '{collection_name}' 中找到 {len(search_results)} 个相关结果",
                results=search_results,
                total_count=len(search_results)
            )
            
        except Exception as e:
            logger.error(f"搜索知识库失败: {e}", exc_info=True)
            return QueryResult(
                success=False,
                message=f"搜索失败: {str(e)}",
                error=str(e)
            )
    
    async def delete_documents(
        self,
        collection_name: str,
        doc_ids: List[str],
    ) -> StorageResult:
        """
        从知识库中删除指定文档
        
        Args:
            collection_name: 知识库名称
            doc_ids: 要删除的文档ID列表
            
        Returns:
            StorageResult: 删除结果
        """
        try:
            if not doc_ids:
                return StorageResult(
                    success=False,
                    message="文档ID列表不能为空",
                    error="Empty document ID list"
                )
            
            # 检查知识库是否存在
            if not await self.vector_db.collection_exists(collection_name):
                return StorageResult(
                    success=False,
                    message=f"知识库 '{collection_name}' 不存在",
                    error="Collection not found"
                )
            
            # 执行删除
            success = await self.vector_db.delete_documents(collection_name, doc_ids)
            
            if success:
                return StorageResult(
                    success=True,
                    message=f"成功从知识库 '{collection_name}' 删除 {len(doc_ids)} 个文档",
                    doc_ids=doc_ids
                )
            else:
                return StorageResult(
                    success=False,
                    message="删除文档失败",
                    error="Delete operation failed"
                )
                
        except Exception as e:
            logger.error(f"删除文档失败: {e}", exc_info=True)
            return StorageResult(
                success=False,
                message=f"删除文档失败: {str(e)}",
                error=str(e)
            )
    
    async def update_document(
        self,
        collection_name: str,
        doc_id: str,
        new_content: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
        options: Optional[StorageOptions] = None,
    ) -> StorageResult:
        """
        更新知识库中的文档
        
        Args:
            collection_name: 知识库名称
            doc_id: 文档ID
            new_content: 新的文本内容
            new_metadata: 新的元数据
            options: 存储选项 (用于内容更新时的分块等处理)
            
        Returns:
            StorageResult: 更新结果
        """
        try:
            # 检查知识库是否存在
            if not await self.vector_db.collection_exists(collection_name):
                return StorageResult(
                    success=False,
                    message=f"知识库 '{collection_name}' 不存在",
                    error="Collection not found"
                )
            
            # 准备更新的元数据
            updated_metadata = None
            if new_metadata:
                updated_metadata = DocumentMetadata(
                    updated_at=time.time(),
                    custom_fields=new_metadata
                )
            
            # 执行更新
            success = await self.vector_db.update_document(
                collection_name=collection_name,
                doc_id=doc_id,
                content=new_content,
                metadata=updated_metadata
            )
            
            if success:
                return StorageResult(
                    success=True,
                    message=f"成功更新知识库 '{collection_name}' 中的文档 {doc_id}",
                    doc_ids=[doc_id]
                )
            else:
                return StorageResult(
                    success=False,
                    message="更新文档失败",
                    error="Update operation failed"
                )
                
        except Exception as e:
            logger.error(f"更新文档失败: {e}", exc_info=True)
            return StorageResult(
                success=False,
                message=f"更新文档失败: {str(e)}",
                error=str(e)
            )
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """
        列出所有知识库及其统计信息
        
        Returns:
            List[Dict]: 知识库信息列表
        """
        try:
            collections = await self.vector_db.list_collections()
            result = []
            
            for collection_name in collections:
                doc_count = await self.vector_db.count_documents(collection_name)
                result.append({
                    "name": collection_name,
                    "document_count": doc_count,
                    "exists": True
                })
            
            return result
            
        except Exception as e:
            logger.error(f"列出知识库失败: {e}", exc_info=True)
            return []
    
    async def create_collection(self, collection_name: str) -> StorageResult:
        """
        创建新的知识库
        
        Args:
            collection_name: 知识库名称
            
        Returns:
            StorageResult: 创建结果
        """
        try:
            if await self.vector_db.collection_exists(collection_name):
                return StorageResult(
                    success=False,
                    message=f"知识库 '{collection_name}' 已存在",
                    error="Collection already exists"
                )
            
            await self.vector_db.create_collection(collection_name)
            
            return StorageResult(
                success=True,
                message=f"成功创建知识库 '{collection_name}'"
            )
            
        except Exception as e:
            logger.error(f"创建知识库失败: {e}", exc_info=True)
            return StorageResult(
                success=False,
                message=f"创建知识库失败: {str(e)}",
                error=str(e)
            )
    
    async def delete_collection(self, collection_name: str) -> StorageResult:
        """
        删除知识库
        
        Args:
            collection_name: 知识库名称
            
        Returns:
            StorageResult: 删除结果
        """
        try:
            if not await self.vector_db.collection_exists(collection_name):
                return StorageResult(
                    success=False,
                    message=f"知识库 '{collection_name}' 不存在",
                    error="Collection not found"
                )
            
            success = await self.vector_db.delete_collection(collection_name)
            
            if success:
                return StorageResult(
                    success=True,
                    message=f"成功删除知识库 '{collection_name}'"
                )
            else:
                return StorageResult(
                    success=False,
                    message="删除知识库失败",
                    error="Delete collection operation failed"
                )
                
        except Exception as e:
            logger.error(f"删除知识库失败: {e}", exc_info=True)
            return StorageResult(
                success=False,
                message=f"删除知识库失败: {str(e)}",
                error=str(e)
            )
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Args:
            collection_name: 知识库名称
            
        Returns:
            Dict: 统计信息
        """
        try:
            if not await self.vector_db.collection_exists(collection_name):
                return {
                    "exists": False,
                    "error": "Collection not found"
                }
            
            doc_count = await self.vector_db.count_documents(collection_name)
            
            return {
                "exists": True,
                "name": collection_name,
                "document_count": doc_count,
                "last_updated": time.time()
            }
            
        except Exception as e:
            logger.error(f"获取知识库统计信息失败: {e}", exc_info=True)
            return {
                "exists": False,
                "error": str(e)
            }
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """
        提取文本关键词 (简单实现，可以后续优化)
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 关键词列表
        """
        try:
            # 这里可以集成更复杂的关键词提取算法
            # 目前使用简单的分词 + 过滤
            import jieba
            
            words = jieba.cut(text)
            # 过滤停用词和短词
            keywords = [
                word.strip() for word in words 
                if len(word.strip()) > 1 and word.strip() not in {'的', '是', '在', '了', '和', '与'}
            ]
            
            # 去重并限制数量
            keywords = list(set(keywords))[:10]
            return keywords
            
        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
            return []


class KnowledgeBaseStorageAPI:
    """
    知识库存储API统一接口
    通过StarMetadata获取插件实例并调用存储功能
    """
    
    @staticmethod
    def get_plugin_instance(context, plugin_name: str = "astrbot_plugin_knowledge_base"):
        """
        通过AstrBot上下文获取知识库插件实例
        
        Args:
            context: AstrBot Context实例
            plugin_name: 插件名称
            
        Returns:
            插件实例或None
        """
        try:
            # 获取所有插件
            all_stars = context.get_all_stars()
            
            # 查找知识库插件
            for star_metadata in all_stars:
                if star_metadata.name == plugin_name and star_metadata.activated:
                    return star_metadata.star_cls
            
            logger.warning(f"未找到激活的插件: {plugin_name}")
            return None
            
        except Exception as e:
            logger.error(f"获取插件实例失败: {e}", exc_info=True)
            return None
    
    @staticmethod
    async def get_storage_api(context, plugin_name: str = "astrbot_plugin_knowledge_base") -> Optional[StorageAPIService]:
        """
        获取存储API服务实例
        
        Args:
            context: AstrBot Context实例
            plugin_name: 插件名称
            
        Returns:
            StorageAPIService实例或None
        """
        try:
            # 获取插件实例
            plugin_instance = KnowledgeBaseStorageAPI.get_plugin_instance(context, plugin_name)
            if not plugin_instance:
                return None
            
            # 确保插件已初始化
            if not await plugin_instance._ensure_initialized():
                logger.error("知识库插件未正确初始化")
                return None
            
            # 创建存储API服务
            storage_api = StorageAPIService(
                vector_db=plugin_instance.kb_service.vector_db,
                kb_service=plugin_instance.kb_service,
                text_splitter=plugin_instance.document_service.text_splitter,
                file_parser=plugin_instance.document_service.file_parser,
                settings=plugin_instance.plugin_config
            )
            
            return storage_api
            
        except Exception as e:
            logger.error(f"获取存储API服务失败: {e}", exc_info=True)
            return None