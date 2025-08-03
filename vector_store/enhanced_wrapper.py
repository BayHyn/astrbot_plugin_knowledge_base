"""
向后兼容包装器
将新的增强型存储无缝集成到现有系统中
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple

from .base import VectorDBBase, Document, DocumentMetadata, Filter, SearchResult
from .enhanced_faiss_store import EnhancedFaissStore
from .keyword_index import KeywordIndex
from .rerank_service import EnhancedHybridReranker
from .api_rerank_config import APIRerankConfig
from .migration_tool import MigrationTool
from ..utils.embedding import EmbeddingUtil
from astrbot.api import logger

class EnhancedVectorStore(VectorDBBase):
    """
    增强型向量存储的向后兼容包装器
    自动处理格式检测和迁移
    """

    def __init__(
        self,
        embedding_util: EmbeddingUtil,
        data_path: str,
        rerank_config: Optional[Dict[str, Any]] = None,
        user_prefs_handler=None,
        max_collections_in_memory: int = 3,
        enable_memory_monitoring: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(embedding_util, data_path)

        # 从配置中读取参数
        memory_config = config.get("memory_optimization", {}) if config else {}
        performance_config = config.get("performance_tuning", {}) if config else {}
        
        # 应用配置值，命令行参数优先
        max_collections_in_memory = memory_config.get("max_collections_in_memory", max_collections_in_memory)
        enable_memory_monitoring = memory_config.get("enable_memory_monitoring", enable_memory_monitoring)
        memory_threshold_mb = memory_config.get("memory_threshold_mb", 1024)
        auto_evict_on_memory_pressure = memory_config.get("auto_evict_on_memory_pressure", True)
        
        # 初始化组件
        self.enhanced_store = EnhancedFaissStore(
            embedding_util, 
            data_path, 
            user_prefs_handler=user_prefs_handler,
            max_collections_in_memory=max_collections_in_memory
        )
        self.migration_tool = MigrationTool(data_path)
        self.reranker = EnhancedHybridReranker(rerank_config)

        # 配置
        self.auto_migrate = True
        self.use_enhanced_search = performance_config.get("enable_hybrid_search", True)
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_threshold_mb = memory_threshold_mb
        self.auto_evict_on_memory_pressure = auto_evict_on_memory_pressure
        
        # 内存监控（如果启用）
        self.memory_monitor = None
        if enable_memory_monitoring:
            try:
                import psutil
                self.memory_monitor = MemoryMonitor(
                    max_memory_mb=memory_threshold_mb,
                    check_interval=60
                )
                logger.info("内存监控已启用")
            except ImportError:
                logger.warning("psutil未安装，内存监控已禁用")
                self.enable_memory_monitoring = False

    async def initialize(self):
        """初始化存储"""
        logger.info("初始化增强型向量存储...")

        # 初始化重排序器
        await self.reranker.initialize()

        # 自动迁移检测
        if self.auto_migrate:
            old_collections = self.migration_tool.detect_old_format()
            if old_collections:
                logger.info(
                    f"检测到 {len(old_collections)} 个旧格式集合，开始自动迁移"
                )
                results = self.migration_tool.migrate_all(self.embedding_util)

                for collection, success in results.items():
                    if success:
                        logger.info(f"集合 {collection} 迁移成功")
                    else:
                        logger.error(f"集合 {collection} 迁移失败")

        logger.info("增强型向量存储初始化完成")

    async def create_collection(self, collection_name: str):
        """创建集合（总是使用新格式）"""
        await self.enhanced_store.create_collection(collection_name)

    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        return await self.enhanced_store.collection_exists(collection_name)

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """添加文档（总是使用新格式）"""
        return await self.enhanced_store.add_documents(collection_name, documents)

    async def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Filter] = None,
    ) -> List[SearchResult]:
        """搜索文档（支持混合搜索和重排序）"""
        # 内存压力检查
        if self.enable_memory_monitoring and self.memory_monitor and self.auto_evict_on_memory_pressure:
            if self.memory_monitor.should_evict_cache():
                logger.info("检测到内存压力，触发缓存清理")
                # 触发enhanced_store的缓存清理
                await self.enhanced_store.collection_cache._evict_lru()
        
        if not self.use_enhanced_search:
            # 回退到基础搜索
            return await self.enhanced_store.search(
                collection_name, query_text, top_k, filters
            )

        # 混合搜索
        return await self._hybrid_search(collection_name, query_text, top_k, filters)

    async def _hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Filter] = None,
    ) -> List[SearchResult]:
        """优化的混合搜索：并发执行向量搜索和关键词搜索"""
        try:
            # 并发执行向量搜索和关键词搜索
            vector_task = asyncio.create_task(
                self.enhanced_store.search(collection_name, query_text, top_k * 2, filters)
            )
            keyword_task = asyncio.create_task(
                self._keyword_search(collection_name, query_text, top_k * 2)
            )
            
            # 等待两个搜索完成
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task, return_exceptions=True
            )
            
            # 处理可能的异常
            if isinstance(vector_results, Exception):
                logger.error(f"向量搜索失败: {vector_results}")
                vector_results = []
            if isinstance(keyword_results, Exception):
                logger.error(f"关键词搜索失败: {keyword_results}")
                keyword_results = []

            if not vector_results and not keyword_results:
                return []

            # 合并结果
            merged_results = self._merge_results(vector_results, keyword_results)

            # 重排序
            reranked_results = await self.reranker.rerank(
                query_text, merged_results, top_k=top_k
            )

            return reranked_results

        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            # 回退到基础向量搜索
            return await self.enhanced_store.search(
                collection_name, query_text, top_k, filters
            )

    async def _keyword_search(
        self, collection_name: str, query_text: str, limit: int = 100
    ) -> List[Tuple[Document, float]]:
        """优化的关键词搜索 - 直接从SQLite获取文档"""
        try:
            # 使用关键词索引进行搜索
            db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
            keyword_index = KeywordIndex(db_path)

            # 执行BM25搜索获取doc_id列表
            keyword_results = keyword_index.search_with_bm25(query_text, limit)

            # 高效批量获取文档 - 直接从SQLite查询而非向量搜索
            documents = []
            if keyword_results:
                # 使用enhanced_store的元数据存储直接获取文档
                with self.enhanced_store.get_metadata_store_for_collection(collection_name) as store:
                    for doc_id, score in keyword_results:
                        try:
                            doc = store.get_document(doc_id)
                            if doc:
                                documents.append((doc, score))
                        except Exception as e:
                            logger.debug(f"获取文档 {doc_id} 失败: {e}")
                            continue

            logger.debug(f"关键词搜索 '{query_text}' 返回 {len(documents)} 个结果")
            return documents

        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []

    def _merge_results(
        self,
        vector_results: List[Tuple[Document, float]] | List[SearchResult],
        keyword_results: List[Tuple[Document, float]],
    ) -> List[SearchResult]:
        """合并向量搜索和关键词搜索结果"""
        # 使用加权平均合并分数
        merged = {}

        # 检查vector_results的类型
        is_vector_search_result = vector_results and isinstance(
            vector_results[0], SearchResult
        )

        # 向量分数
        for item in vector_results:
            if is_vector_search_result:
                doc = item.document
                score = item.score
            else:
                doc, score = item

            merged[doc.id] = {"doc": doc, "vector_score": score, "keyword_score": 0.0}

        # 关键词分数
        for doc, score in keyword_results:
            if doc.id in merged:
                merged[doc.id]["keyword_score"] = score
            else:
                merged[doc.id] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "keyword_score": score,
                }

        # 计算综合分数
        results = []
        for data in merged.values():
            combined_score = 0.7 * data["vector_score"] + 0.3 * data["keyword_score"]
            # 返回SearchResult
            results.append(
                SearchResult(
                    document=data["doc"],
                    score=data["vector_score"],
                    rerank_score=combined_score,
                )
            )

        # 排序
        results.sort(
            key=lambda x: x.rerank_score if x.rerank_score is not None else x.score,
            reverse=True,
        )
        return results

    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        return await self.enhanced_store.delete_collection(collection_name)

    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        return await self.enhanced_store.list_collections()

    async def cleanup_corrupted_collections(self) -> List[str]:
        """清理损坏的集合"""
        return await self.enhanced_store.cleanup_corrupted_collections()

    async def count_documents(self, collection_name: str) -> int:
        """统计文档数量"""
        return await self.enhanced_store.count_documents(collection_name)

    async def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        """从指定集合中删除文档"""
        return await self.enhanced_store.delete_documents(collection_name, doc_ids)

    async def update_document(
        self,
        collection_name: str,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
    ) -> bool:
        """更新指定集合中的文档"""
        return await self.enhanced_store.update_document(
            collection_name, doc_id, content, metadata
        )

    async def close(self):
        """关闭存储"""
        await self.enhanced_store.close()

    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        info = {
            "type": "enhanced_faiss",
            "features": [
                "faiss_vector_index",
                "sqlite_metadata",
                "keyword_index",
                "rerank_support",
                "migration_support",
                "lru_cache",
            ],
            "format": {"metadata": ".enhanced.db", "vectors": ".faiss"},
            "cache_stats": self.enhanced_store.get_cache_stats(),
        }
        
        # 添加内存监控信息
        if self.enable_memory_monitoring and self.memory_monitor:
            memory_info = self.memory_monitor.get_memory_usage()
            info["memory_monitoring"] = {
                "enabled": True,
                "memory_usage": memory_info,
                "threshold_mb": self.memory_threshold_mb,
                "auto_evict_enabled": self.auto_evict_on_memory_pressure,
            }
        else:
            info["memory_monitoring"] = {"enabled": False}
            
        return info

    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        return self.migration_tool.get_migration_status()

    async def trigger_migration(self, force: bool = False) -> Dict[str, bool]:
        """手动触发迁移"""
        return self.migration_tool.migrate_all(self.embedding_util, force)

    def cleanup_old_files(self, collection_name: str) -> bool:
        """清理旧文件"""
        return self.migration_tool.cleanup_old_files(collection_name)


# 工厂函数
def create_enhanced_store(
    embedding_util: EmbeddingUtil, 
    data_path: str, 
    use_enhanced: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> VectorDBBase:
    """
    创建增强型存储实例

    Args:
        embedding_util: 嵌入工具
        data_path: 数据路径
        use_enhanced: 是否使用增强功能
        config: 完整配置字典（包含memory_optimization等）

    Returns:
        VectorDBBase: 向量存储实例
    """
    if use_enhanced:
        return EnhancedVectorStore(embedding_util, data_path, config=config)
    else:
        # 回退到旧格式
        from .faiss_store import FaissStore
        return FaissStore(embedding_util, data_path)


# 配置管理（更新版）
class EnhancedStoreConfig:
    """增强存储配置（支持API重排序）"""

    def __init__(self):
        self.auto_migrate = True
        self.use_enhanced_search = True
        self.rerank_strategy = "auto"  # auto, api, simple
        self.rerank_config = {
            "strategy": "auto",
            "api": {
                "provider": "cohere",
                "api_key": "",
                "timeout": 30,
                "max_retries": 3,
                "enable_cache": True,
                "cache_ttl": 3600,
                "fallback_strategy": "simple",
            },
        }
        self.migration_config = {
            "backup_before_migrate": True,
            "validate_after_migrate": True,
            "cleanup_old_files": False,
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "auto_migrate": self.auto_migrate,
            "use_enhanced_search": self.use_enhanced_search,
            "rerank_strategy": self.rerank_strategy,
            "rerank_config": self.rerank_config,
            "migration_config": self.migration_config,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnhancedStoreConfig":
        """从字典创建配置"""
        config = cls()
        config.auto_migrate = config_dict.get("auto_migrate", True)
        config.use_enhanced_search = config_dict.get("use_enhanced_search", True)
        config.rerank_strategy = config_dict.get("rerank_strategy", "auto")

        if "rerank_config" in config_dict:
            config.rerank_config.update(config_dict["rerank_config"])

        if "migration_config" in config_dict:
            config.migration_config.update(config_dict["migration_config"])

        return config


class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self, max_memory_mb: int = 1024, check_interval: int = 60):
        try:
            import psutil
            self.psutil = psutil
            self.max_memory_mb = max_memory_mb
            self.check_interval = check_interval
            self.last_check = 0
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("psutil未安装，内存监控不可用")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        if not self.available:
            return {"available": False}
            
        try:
            process = self.psutil.Process()
            memory_info = process.memory_info()
            return {
                "available": True,
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "system_available_mb": self.psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"获取内存使用情况失败: {e}")
            return {"available": False, "error": str(e)}
            
    def should_evict_cache(self) -> bool:
        """检查是否应该驱逐缓存"""
        if not self.available:
            return False
            
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return False
            
        self.last_check = current_time
        memory_info = self.get_memory_usage()
        
        if memory_info.get("available"):
            return memory_info.get("rss_mb", 0) > self.max_memory_mb
        return False
        
    def log_memory_stats(self):
        """记录内存统计信息"""
        if not self.available:
            return
            
        memory_info = self.get_memory_usage()
        if memory_info.get("available"):
            logger.info(
                f"内存使用: RSS={memory_info['rss_mb']:.1f}MB, "
                f"VMS={memory_info['vms_mb']:.1f}MB, "
                f"进程占用={memory_info['percent']:.1f}%, "
                f"系统可用={memory_info['system_available_mb']:.1f}MB"
            )


# 工厂函数增强
def create_enhanced_store(
    embedding_util: EmbeddingUtil, 
    data_path: str, 
    use_enhanced: bool = True,
    max_collections_in_memory: int = 3,
    enable_memory_monitoring: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> VectorDBBase:
    """
    创建增强型存储实例

    Args:
        embedding_util: 嵌入工具
        data_path: 数据路径
        use_enhanced: 是否使用增强功能
        max_collections_in_memory: 内存中最大集合数
        enable_memory_monitoring: 是否启用内存监控
        config: 完整配置字典（优先级高于直接参数）

    Returns:
        VectorDBBase: 向量存储实例
    """
    if use_enhanced:
        return EnhancedVectorStore(
            embedding_util, 
            data_path,
            max_collections_in_memory=max_collections_in_memory,
            enable_memory_monitoring=enable_memory_monitoring,
            config=config
        )
    else:
        # 回退到旧格式
        from .faiss_store import FaissStore
        return FaissStore(embedding_util, data_path)
