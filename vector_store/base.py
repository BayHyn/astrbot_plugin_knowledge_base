from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from ..utils.embedding import EmbeddingSolutionHelper

DEFAULT_BATCH_SIZE = 10  # 默认批处理大小
MAX_RETRIES = 3  # 最大重试次数


@dataclass
class DocumentMetadata:
    """文档元数据结构，用于存储与文档相关的结构化信息"""

    source: Optional[str] = None  # 文档来源，例如文件路径或URL
    created_at: Optional[float] = None  # 文档创建时间戳
    updated_at: Optional[float] = None  # 文档更新时间戳
    # 可以根据需要添加更多通用字段
    # 例如: author: Optional[str] = None
    # 为了保持灵活性，允许存储自定义键值对
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    text_content: str
    embedding: Optional[List[float]] = None  # 向量数据，添加时生成，查询时不需要
    metadata: DocumentMetadata = field(
        default_factory=DocumentMetadata
    )  # 结构化的文档元数据
    id: Optional[str] = None  # 文档在向量数据库中的唯一 ID


@dataclass
class FilterCondition:
    """过滤条件，用于在搜索时根据元数据进行筛选"""

    key: str  # 元数据字段名，例如 "source" 或 "custom_fields.category"
    operator: str  # 比较操作符，例如 "=", "!=", ">", "<", "in", "contains"
    value: Any  # 比较值


@dataclass
class Filter:
    """过滤器，包含一个或多个过滤条件"""

    conditions: List[FilterCondition] = field(default_factory=list)
    logic: str = "and"  # 条件之间的逻辑关系，"and" 或 "or"


@dataclass
class SearchResult:
    """搜索结果，包含文档和相关分数"""

    document: Document
    score: float  # 向量相似度分数
    rerank_score: Optional[float] = None  # 重排序后的分数（如果适用）


@dataclass
class ProcessingBatch:
    documents: List[Document]
    retry_count: int = 0  # 记录当前批次的重试次数


class VectorDBBase(ABC):
    def __init__(self, embedding_util: EmbeddingSolutionHelper, data_path: str):
        self.embedding_util = embedding_util  # EmbeddingUtil 实例
        self.data_path = data_path  # 数据存储路径 (主要用于 Faiss Lite)

    @abstractmethod
    async def initialize(self):
        """初始化数据库连接和集合等"""
        pass

    @abstractmethod
    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """向指定集合添加文档，返回文档ID列表"""
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Filter] = None,
    ) -> List[SearchResult]:
        """
        在指定集合中搜索相关文档

        Args:
            collection_name: 集合名称
            query_text: 查询文本
            top_k: 返回的最相似文档数量
            filters: 可选的元数据过滤器

        Returns:
            搜索结果列表，按相关性排序
        """
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str):
        """创建集合"""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合及其所有数据"""
        pass

    @abstractmethod
    async def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        """
        从指定集合中删除文档

        Args:
            collection_name: 集合名称
            doc_ids: 要删除的文档ID列表

        Returns:
            如果至少删除了一个文档则返回True，否则返回False
        """
        pass

    @abstractmethod
    async def update_document(
        self,
        collection_name: str,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
    ) -> bool:
        """
        更新指定集合中的文档

        Args:
            collection_name: 集合名称
            doc_id: 要更新的文档ID
            content: 新的文本内容（可选）
            metadata: 新的元数据（可选）

        Returns:
            如果更新成功则返回True，否则返回False
        """
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        pass

    @abstractmethod
    async def count_documents(self, collection_name: str) -> int:
        """计算集合中的文档数量"""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        pass

    @abstractmethod
    async def close(self):
        """关闭数据库连接"""
        pass
