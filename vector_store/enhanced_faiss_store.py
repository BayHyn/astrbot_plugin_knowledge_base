"""
增强型FAISS + SQLite向量存储实现
提供关键词索引、知识图谱接口和重排序功能
优化存储效率，支持向后兼容
"""

import os
import sqlite3
import json
import pickle
import asyncio
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from contextlib import contextmanager

from .base import (
    VectorDBBase,
    Document,
    DocumentMetadata,
    Filter,
    SearchResult,
    ProcessingBatch,
    DEFAULT_BATCH_SIZE,
    MAX_RETRIES,
)
from ..utils.embedding import EmbeddingSolutionHelper
from astrbot.api import logger


# 存储优化配置
@dataclass
class StorageConfig:
    """存储配置参数"""

    use_compression: bool = True  # 是否启用向量压缩
    compression_bits: int = 8  # PQ压缩位数
    use_ivf: bool = True  # 是否使用IVF索引
    nlist: int = 100  # IVF聚类中心数
    use_mmap: bool = True  # 是否使用内存映射
    cache_size: int = 1000  # 缓存大小
    batch_size: int = 100  # 批处理大小


# 增强的文档结构
@dataclass
class EnhancedDocument(Document):
    """增强文档结构，支持更多元数据"""

    keywords: List[str] = field(default_factory=list)  # 关键词列表
    doc_hash: str = ""  # 文档哈希
    created_at: float = 0.0  # 创建时间戳
    updated_at: float = 0.0  # 更新时间戳
    graph_nodes: List[str] = field(default_factory=list)  # 知识图谱节点
    graph_edges: List[Tuple[str, str, str]] = field(default_factory=list)  # 知识图谱边


class SQLiteMetadataStore:
    """SQLite元数据存储层"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            # 文档表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    text_content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    keywords TEXT,
                    doc_hash TEXT,
                    created_at REAL,
                    updated_at REAL,
                    vector_id INTEGER
                )
            """)

            # 关键词倒排索引表
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts USING fts5(
                    doc_id,
                    keywords,
                    text_content,
                    tokenize='porter'
                )
            """)

            # 知识图谱节点表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT UNIQUE NOT NULL,
                    node_type TEXT,
                    properties TEXT,
                    created_at REAL
                )
            """)

            # 知识图谱边表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_node TEXT NOT NULL,
                    target_node TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    created_at REAL,
                    UNIQUE(source_node, target_node, relation_type)
                )
            """)

            # 索引优化
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vector_id ON documents(vector_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_id ON graph_nodes(node_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_nodes ON graph_edges(source_node, target_node)"
            )

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接（线程安全）"""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        yield self._local.connection

    def add_document(self, doc: EnhancedDocument, vector_id: int) -> int:
        """添加文档到数据库"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO documents 
                (doc_id, text_content, embedding, metadata, keywords, doc_hash, 
                 created_at, updated_at, vector_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    doc.id,
                    doc.text_content,
                    pickle.dumps(doc.embedding) if doc.embedding else None,
                    json.dumps(doc.metadata),
                    json.dumps(doc.keywords),
                    doc.doc_hash,
                    doc.created_at,
                    doc.updated_at,
                    vector_id,
                ),
            )

            # 更新FTS索引
            conn.execute(
                """
                INSERT OR REPLACE INTO keywords_fts (doc_id, keywords, text_content)
                VALUES (?, ?, ?)
            """,
                (doc.id, " ".join(doc.keywords), doc.text_content),
            )

            conn.commit()
            return cursor.lastrowid

    def get_document(self, doc_id: str) -> Optional[EnhancedDocument]:
        """根据文档ID获取文档"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                return EnhancedDocument(
                    text_content=row["text_content"],
                    embedding=pickle.loads(row["embedding"])
                    if row["embedding"]
                    else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    id=row["doc_id"],
                    keywords=json.loads(row["keywords"]) if row["keywords"] else [],
                    doc_hash=row["doc_hash"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        return None

    def search_by_keywords(
        self, query: str, limit: int = 100
    ) -> List[Tuple[str, float]]:
        """基于关键词的全文搜索"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT doc_id, rank FROM keywords_fts 
                WHERE keywords_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """,
                (query, limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append((row["doc_id"], row["rank"]))
            return results

    def get_all_documents(self) -> List[Tuple[str, int]]:
        """获取所有文档ID和对应的向量ID"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT doc_id, vector_id FROM documents ORDER BY vector_id"
            )
            return [(row["doc_id"], row["vector_id"]) for row in cursor.fetchall()]

    def count_documents(self) -> int:
        """统计文档数量"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM documents")
            return cursor.fetchone()["count"]

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM keywords_fts WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_documents_by_vector_ids(
        self, vector_ids: List[int], filters: Optional[Filter] = None
    ) -> List[EnhancedDocument]:
        """根据向量ID列表获取文档，可选地应用过滤器"""
        if not vector_ids:
            return []

        placeholders = ",".join("?" * len(vector_ids))
        base_query = f"SELECT * FROM documents WHERE vector_id IN ({placeholders})"
        params = list(vector_ids)

        if filters:
            filter_clause, filter_params = self._build_filter_clause(filters)
            if filter_clause:
                base_query += f" AND {filter_clause}"
                params.extend(filter_params)

        with self._get_connection() as conn:
            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                doc = EnhancedDocument(
                    text_content=row["text_content"],
                    embedding=pickle.loads(row["embedding"])
                    if row["embedding"]
                    else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    id=row["doc_id"],
                    keywords=json.loads(row["keywords"]) if row["keywords"] else [],
                    doc_hash=row["doc_hash"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                documents.append(doc)

            return documents

    def _build_filter_clause(self, filters: Filter) -> Tuple[str, List[Any]]:
        """构建SQL过滤子句"""
        if not filters.conditions:
            return "", []

        clauses = []
        params = []

        for condition in filters.conditions:
            # 简化处理，假设所有字段都在metadata JSON字段中
            # 实际应用中可能需要更复杂的逻辑来处理不同类型的字段
            if condition.key.startswith("metadata."):
                json_key = condition.key.split(".", 1)[1]
                if condition.operator == "=":
                    clauses.append(f"json_extract(metadata, '$.{json_key}') = ?")
                    params.append(condition.value)
                elif condition.operator == "!=":
                    clauses.append(f"json_extract(metadata, '$.{json_key}') != ?")
                    params.append(condition.value)
                elif condition.operator == "in":
                    if isinstance(condition.value, (list, tuple)):
                        in_placeholders = ",".join("?" * len(condition.value))
                        clauses.append(
                            f"json_extract(metadata, '$.{json_key}') IN ({in_placeholders})"
                        )
                        params.extend(condition.value)
                # 可以添加更多操作符...

            # 处理直接字段（如 created_at）
            elif condition.key in ["created_at", "updated_at"]:
                if condition.operator in ["=", "!=", ">", "<", ">=", "<="]:
                    clauses.append(f"{condition.key} {condition.operator} ?")
                    params.append(condition.value)

            # 处理 source 字段
            elif condition.key == "source":
                # 假设source存储在metadata的'source'字段中
                if condition.operator == "=":
                    clauses.append("json_extract(metadata, '$.source') = ?")
                    params.append(condition.value)
                elif condition.operator == "!=":
                    clauses.append("json_extract(metadata, '$.source') != ?")
                    params.append(condition.value)
                elif condition.operator == "in":
                    if isinstance(condition.value, (list, tuple)):
                        in_placeholders = ",".join("?" * len(condition.value))
                        clauses.append(
                            f"json_extract(metadata, '$.source') IN ({in_placeholders})"
                        )
                        params.extend(condition.value)

            # 可以添加对其他字段的处理...

        if not clauses:
            return "", []

        logic_op = " AND " if filters.logic.lower() == "and" else " OR "
        return logic_op.join(clauses), params

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection


class FAISSIndexStore:
    """FAISS向量索引存储层"""

    def __init__(self, index_path: str, config: StorageConfig):
        self.index_path = index_path
        self.config = config
        self.index = None
        self.dimension = None
        self.is_trained = False

    def initialize(self, dimension: int):
        """初始化FAISS索引"""
        self.dimension = dimension

        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.create_index()

    def create_index(self):
        """创建新的FAISS索引"""
        if self.config.use_compression:
            # 使用Product Quantization压缩
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.config.nlist,
                self.config.compression_bits,
                8,
            )
        elif self.config.use_ivf:
            # 使用IVF索引
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.config.nlist
            )
        else:
            # 使用基础L2索引
            self.index = faiss.IndexFlatL2(self.dimension)

        self.is_trained = not (self.config.use_compression or self.config.use_ivf)

    def load_index(self):
        """加载FAISS索引"""
        try:
            if self.config.use_mmap:
                self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            else:
                self.index = faiss.read_index(self.index_path)
            self.is_trained = True
            logger.info(f"成功加载FAISS索引: {self.index_path}")
        except Exception as e:
            logger.error(f"加载FAISS索引失败: {e}")
            self.create_index()

    def save_index(self):
        """保存FAISS索引"""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"成功保存FAISS索引: {self.index_path}")

    def add_vectors(self, vectors: np.ndarray) -> int:
        """添加向量到索引"""
        if self.index is None:
            raise ValueError("索引未初始化")

        if not self.is_trained and (self.config.use_compression or self.config.use_ivf):
            # 训练索引
            self.index.train(vectors)
            self.is_trained = True

        start_id = self.index.ntotal
        self.index.add(vectors)
        return start_id

    def search_vectors(
        self, query_vectors: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """搜索相似向量"""
        if self.index is None:
            raise ValueError("索引未初始化")

        if self.index.ntotal == 0:
            return np.array([[]]), np.array([[]])

        distances, indices = self.index.search(query_vectors, min(k, self.index.ntotal))
        return distances, indices

    def count_vectors(self) -> int:
        """统计向量数量"""
        return self.index.ntotal if self.index else 0

    def close(self):
        """关闭索引"""
        self.save_index()
        self.index = None


class EnhancedFaissStore(VectorDBBase):
    """增强型FAISS + SQLite向量存储实现"""

    def __init__(
        self,
        embedding_util: EmbeddingSolutionHelper,
        data_path: str,
        config: Optional[StorageConfig] = None,
    ):
        super().__init__(embedding_util, data_path)
        self.config = config or StorageConfig()

        # 创建数据目录
        os.makedirs(data_path, exist_ok=True)

        # 初始化存储组件
        self.metadata_store = None
        self.index_store = None
        self.collection_name = None

        # 缓存和锁
        self._cache = {}
        self._lock = asyncio.Lock()

    async def initialize(self):
        """初始化存储"""
        logger.info("初始化增强型FAISS存储...")

    async def create_collection(self, collection_name: str):
        """创建集合"""
        async with self._lock:
            if await self.collection_exists(collection_name):
                logger.info(f"集合 '{collection_name}' 已存在")
                return

            # 初始化存储组件
            db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
            index_path = os.path.join(self.data_path, f"{collection_name}.faiss")

            self.metadata_store = SQLiteMetadataStore(db_path)
            self.index_store = FAISSIndexStore(index_path, self.config)

            # 获取维度并初始化索引
            dimension = self.embedding_util.get_dimensions(collection_name)
            self.index_store.initialize(dimension)

            self.collection_name = collection_name
            logger.info(f"成功创建集合: {collection_name}")

    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        index_path = os.path.join(self.data_path, f"{collection_name}.faiss")
        return os.path.exists(db_path) and os.path.exists(index_path)

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """添加文档到集合"""
        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name)

        if not documents:
            return []

        # 准备增强文档
        enhanced_docs = []
        for doc in documents:
            enhanced_doc = EnhancedDocument(
                text_content=doc.text_content,
                embedding=doc.embedding,
                metadata=doc.metadata,
                id=doc.id or "",
                keywords=self._extract_keywords(doc.text_content),
                doc_hash=self._compute_hash(doc.text_content),
                created_at=time.time(),
                updated_at=time.time(),
            )
            enhanced_docs.append(enhanced_doc)

        # 生成向量
        texts = [doc.text_content for doc in enhanced_docs]
        embeddings = await self.embedding_util.get_embeddings_async(
            texts, collection_name
        )

        # 过滤有效向量
        valid_docs = []
        valid_embeddings = []
        for doc, embedding in zip(enhanced_docs, embeddings):
            if embedding is not None:
                doc.embedding = embedding
                valid_docs.append(doc)
                valid_embeddings.append(embedding)

        if not valid_docs:
            logger.warning("没有有效的向量可以添加")
            return []

        # 添加到FAISS索引
        vectors = np.array(valid_embeddings).astype("float32")
        faiss.normalize_L2(vectors)

        start_id = self.index_store.add_vectors(vectors)

        # 添加到SQLite
        doc_ids = []
        for i, doc in enumerate(valid_docs):
            doc.id = f"{collection_name}_{start_id + i}"
            self.metadata_store.add_document(doc, start_id + i)
            doc_ids.append(doc.id)

        # 保存索引
        self.index_store.save_index()

        logger.info(f"成功添加 {len(doc_ids)} 个文档到集合 '{collection_name}'")
        return doc_ids

    async def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Filter] = None,
    ) -> List[SearchResult]:
        """搜索相似文档"""
        if not await self.collection_exists(collection_name):
            logger.warning(f"集合 '{collection_name}' 不存在")
            return []

        # 获取查询向量
        query_embedding = await self.embedding_util.get_embedding_async(
            query_text, collection_name
        )
        if query_embedding is None:
            logger.error("无法生成查询向量")
            return []

        # 向量搜索
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)

        distances, indices = self.index_store.search_vectors(query_vector, top_k * 2)

        # 获取文档
        vector_ids = [idx for idx in indices[0] if idx != -1]  # 过滤有效索引
        if not vector_ids:
            return []

        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        metadata_store = SQLiteMetadataStore(db_path)
        docs = metadata_store.get_documents_by_vector_ids(vector_ids, filters)
        metadata_store.close()

        # 构建搜索结果
        results = []
        doc_dict = {doc.vector_id: doc for doc in docs}  # 通过vector_id快速查找文档

        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in doc_dict:  # 有效索引且文档存在（通过过滤器）
                doc = doc_dict[idx]
                similarity = 1.0 - (distance / 2.0)  # 转换为相似度
                results.append(SearchResult(document=doc, score=float(similarity)))

        # 按相似度排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        """从指定集合中删除文档"""
        if not await self.collection_exists(collection_name):
            logger.warning(f"集合 '{collection_name}' 不存在")
            return False

        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        metadata_store = SQLiteMetadataStore(db_path)

        deleted_count = 0
        for doc_id in doc_ids:
            if metadata_store.delete_document(doc_id):
                deleted_count += 1

        metadata_store.close()

        logger.info(f"从集合 '{collection_name}' 中删除了 {deleted_count} 个文档")
        return deleted_count > 0

    async def update_document(
        self,
        collection_name: str,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
    ) -> bool:
        """更新指定集合中的文档"""
        if not await self.collection_exists(collection_name):
            logger.warning(f"集合 '{collection_name}' 不存在")
            return False

        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        metadata_store = SQLiteMetadataStore(db_path)

        # 获取现有文档
        existing_doc = metadata_store.get_document(doc_id)
        if not existing_doc:
            logger.warning(f"文档 '{doc_id}' 不存在于集合 '{collection_name}' 中")
            metadata_store.close()
            return False

        # 更新内容
        if content is not None:
            existing_doc.text_content = content
            # 如果内容改变，需要重新生成embedding和keywords
            existing_doc.embedding = None  # 标记为需要重新生成
            existing_doc.keywords = self._extract_keywords(content)
            existing_doc.doc_hash = self._compute_hash(content)

        # 更新元数据
        if metadata is not None:
            existing_doc.metadata = metadata

        # 更新时间戳
        existing_doc.updated_at = time.time()

        # 如果内容被更新，重新生成embedding
        if content is not None:
            new_embedding = await self.embedding_util.get_embedding_async(
                content, collection_name
            )
            if new_embedding is not None:
                existing_doc.embedding = new_embedding
            else:
                logger.error(f"无法为文档 '{doc_id}' 生成新的embedding")
                metadata_store.close()
                return False

        # 保存更新后的文档
        # 注意：这里简化了处理，实际应用中可能需要更复杂的逻辑来处理向量索引的更新
        # 例如，可能需要先从FAISS索引中删除旧向量，再添加新向量
        # 为了保持简单，我们假设向量ID保持不变，只更新元数据
        # 在实际生产环境中，这需要更仔细的设计
        metadata_store.add_document(
            existing_doc, existing_doc.vector_id
        )  # 假设vector_id不变
        metadata_store.close()

        logger.info(f"成功更新文档 '{doc_id}' 在集合 '{collection_name}' 中")
        return True

    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        if not await self.collection_exists(collection_name):
            return False

        try:
            # 删除文件
            db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
            index_path = os.path.join(self.data_path, f"{collection_name}.faiss")

            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(index_path):
                os.remove(index_path)

            logger.info(f"成功删除集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        collections = []
        if not os.path.exists(self.data_path):
            return collections

        for filename in os.listdir(self.data_path):
            if filename.endswith(".enhanced.db"):
                collection_name = filename[:-12]  # 移除.enhanced.db
                collections.append(collection_name)

        return collections

    async def count_documents(self, collection_name: str) -> int:
        """统计文档数量"""
        if not await self.collection_exists(collection_name):
            return 0

        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        metadata_store = SQLiteMetadataStore(db_path)
        count = metadata_store.count_documents()
        metadata_store.close()
        return count

    async def close(self):
        """关闭存储"""
        if self.metadata_store:
            self.metadata_store.close()
        if self.index_store:
            self.index_store.close()
        logger.info("增强型FAISS存储已关闭")

    def _extract_keywords(self, text: str) -> List[str]:
        """使用jieba提取关键词"""
        try:
            import jieba.analyse

            return jieba.analyse.extract_tags(text, topK=20)
        except ImportError:
            logger.warning("jieba库未安装，无法提取关键词。将返回空列表。")
            return []
        except Exception as e:
            logger.error(f"使用jieba提取关键词时出错: {e}")
            return []

    def _compute_hash(self, text: str) -> str:
        """计算文本哈希"""
        import hashlib

        return hashlib.md5(text.encode()).hexdigest()
