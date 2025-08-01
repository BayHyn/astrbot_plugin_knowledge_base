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
from ..utils.embedding import EmbeddingUtil
from astrbot.api import logger


# 存储优化配置
@dataclass
class StorageConfig:
    """存储配置参数"""

    use_compression: bool = False  # 暂时禁用压缩，避免训练数据不足问题
    compression_bits: int = 8  # PQ压缩位数
    use_ivf: bool = False  # 暂时禁用IVF，避免训练数据不足问题
    nlist: int = 10  # IVF聚类中心数（减少到10，降低训练数据需求）
    use_mmap: bool = True  # 是否使用内存映射
    cache_size: int = 1000  # 缓存大小
    batch_size: int = 100  # 批处理大小

    # TODO: 未来优化配置策略
    # - 根据预期数据规模动态调整配置
    # - 实现配置自动优化算法
    # - 支持运行时配置切换


# 增强的文档结构
@dataclass
class EnhancedDocument(Document):
    """增强文档结构，支持更多元数据"""

    keywords: List[str] = field(default_factory=list)  # 关键词列表
    doc_hash: str = ""  # 文档哈希
    created_at: float = 0.0  # 创建时间戳
    updated_at: float = 0.0  # 更新时间戳
    vector_id: int = -1  # 向量ID
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
                    vector_id=row["vector_id"],
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
            logger.debug("vector_ids为空，返回空列表")
            return []

        logger.debug(f"查询向量ID: {vector_ids[:10]}... (共{len(vector_ids)}个)")

        placeholders = ",".join("?" * len(vector_ids))
        base_query = f"SELECT * FROM documents WHERE vector_id IN ({placeholders})"
        params = list(vector_ids)

        if filters:
            filter_clause, filter_params = self._build_filter_clause(filters)
            if filter_clause:
                base_query += f" AND {filter_clause}"
                params.extend(filter_params)

        logger.debug(f"执行SQL查询: {base_query}")
        logger.debug(f"参数: {params[:10]}... (共{len(params)}个)")

        with self._get_connection() as conn:
            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()

            logger.debug(f"SQL查询返回 {len(rows)} 行数据")

            # TODO: 添加数据库一致性检查
            # 如果没有匹配的文档，检查数据库中实际存储的vector_id范围
            if len(rows) == 0 and len(vector_ids) > 0:
                logger.warning("没有找到匹配的文档，检查数据库中的vector_id范围...")
                try:
                    cursor = conn.execute(
                        "SELECT MIN(vector_id), MAX(vector_id), COUNT(*) FROM documents"
                    )
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        min_id, max_id, count = result
                        logger.warning(
                            f"数据库中vector_id范围: {min_id}-{max_id}, 总数: {count}"
                        )
                        logger.warning(
                            f"查询的vector_id范围: {min(vector_ids)}-{max(vector_ids)}"
                        )

                        # 检查是否有任何匹配
                        sample_ids = vector_ids[:5]  # 检查前5个ID
                        for vid in sample_ids:
                            cursor = conn.execute(
                                "SELECT COUNT(*) FROM documents WHERE vector_id = ?",
                                (vid,),
                            )
                            exists = cursor.fetchone()[0]
                            logger.debug(f"vector_id {vid} 存在: {exists > 0}")

                        # 如果数据库中的vector_id都是负数或者从0开始，而查询的是从其他数字开始
                        # 这可能表明数据不一致，需要修复
                        if max_id < min(vector_ids):
                            logger.error(
                                f"数据不一致：数据库vector_id最大值({max_id}) < 查询最小值({min(vector_ids)})"
                            )
                            logger.error("可能需要重建索引或修复数据")
                    else:
                        logger.warning("数据库中没有任何文档记录")

                except Exception as e:
                    logger.error(f"检查数据库状态时出错: {e}")

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
                    vector_id=row["vector_id"],
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
        # TODO: 添加维度验证和索引兼容性检查
        if dimension is None or dimension <= 0:
            raise ValueError(f"无效的维度值: {dimension}")

        self.dimension = dimension
        logger.debug(f"初始化FAISS索引，维度: {dimension}")

        if os.path.exists(self.index_path):
            try:
                self.load_index()
                # 验证加载的索引维度是否匹配
                if self.index is not None and hasattr(self.index, "d"):
                    if self.index.d != dimension:
                        logger.warning(
                            f"索引维度不匹配：文件={self.index.d}, 期望={dimension}，重新创建索引"
                        )
                        self.create_index()
                    else:
                        logger.debug(f"索引加载成功，维度匹配: {dimension}")
                else:
                    logger.warning("加载的索引无效，重新创建")
                    self.create_index()
            except Exception as e:
                logger.error(f"加载索引失败，重新创建: {e}")
                self.create_index()
        else:
            self.create_index()

        # 最终验证索引是否正确初始化
        if self.index is None:
            raise ValueError("FAISS索引初始化失败")

    def create_index(self):
        """创建新的FAISS索引"""
        # TODO: 根据数据量和性能需求动态选择索引类型
        logger.debug(f"创建新的FAISS索引，维度: {self.dimension}")

        try:
            # TODO: 实现智能索引类型选择策略
            # 对于小数据集，使用简单的FlatL2索引避免训练点不足的问题
            # IVF索引需要大量训练数据，通常需要聚类数的至少39倍的训练点

            # 暂时禁用复杂索引类型，使用最稳定的FlatL2索引
            # 这样可以避免训练点不足的问题
            use_simple_index = True  # TODO: 未来根据数据量动态决定

            if use_simple_index:
                # 使用基础L2索引，不需要训练
                logger.info("使用基础FlatL2索引（适合小到中等规模数据集）")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.is_trained = True
            elif self.config.use_compression:
                # 使用Product Quantization压缩
                # 需要足够的训练数据：至少 nlist * 39 个向量
                logger.info("使用IVF+PQ压缩索引")
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFPQ(
                    quantizer,
                    self.dimension,
                    self.config.nlist,
                    self.config.compression_bits,
                    8,
                )
                self.is_trained = False
            elif self.config.use_ivf:
                # 使用IVF索引
                # 需要足够的训练数据：至少 nlist * 39 个向量
                logger.info("使用IVF索引")
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.config.nlist
                )
                self.is_trained = False
            else:
                # 使用基础L2索引
                logger.info("使用基础FlatL2索引")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.is_trained = True

            logger.debug(
                f"索引创建成功，类型: {type(self.index).__name__}, 是否已训练: {self.is_trained}"
            )

        except Exception as e:
            logger.error(f"创建FAISS索引失败: {e}")
            # 如果创建失败，回退到最简单的索引类型
            try:
                logger.info("尝试创建基础L2索引作为备用")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.is_trained = True
                logger.info("备用索引创建成功")
            except Exception as fallback_e:
                logger.error(f"备用索引创建也失败: {fallback_e}")
                raise ValueError(f"无法创建任何类型的FAISS索引: {fallback_e}")

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

        # TODO: 实现更智能的训练策略和错误恢复机制
        if not self.is_trained and (self.config.use_compression or self.config.use_ivf):
            # 检查是否有足够的训练数据
            n_vectors = vectors.shape[0]
            min_required = self.config.nlist * 39  # FAISS建议的最小训练数据量

            if n_vectors < min_required:
                logger.warning(
                    f"训练数据不足：当前 {n_vectors} 个向量，需要至少 {min_required} 个"
                )
                logger.info("降级到基础FlatL2索引以避免训练错误")

                # 重新创建为简单索引
                try:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.is_trained = True
                    logger.info("成功降级到FlatL2索引")
                except Exception as e:
                    logger.error(f"降级索引失败: {e}")
                    raise ValueError(f"无法创建替代索引: {e}")
            else:
                # 有足够的训练数据，进行训练
                try:
                    logger.info(f"开始训练索引，训练数据: {n_vectors} 个向量")
                    self.index.train(vectors)
                    self.is_trained = True
                    logger.info("索引训练完成")
                except Exception as e:
                    logger.error(f"索引训练失败: {e}")
                    # 训练失败，降级到简单索引
                    logger.info("训练失败，降级到基础FlatL2索引")
                    try:
                        self.index = faiss.IndexFlatL2(self.dimension)
                        self.is_trained = True
                        logger.info("成功降级到FlatL2索引")
                    except Exception as fallback_e:
                        logger.error(f"降级索引失败: {fallback_e}")
                        raise ValueError(f"索引训练和降级都失败: {fallback_e}")

        start_id = self.index.ntotal
        try:
            self.index.add(vectors)
            logger.debug(f"成功添加 {vectors.shape[0]} 个向量，起始ID: {start_id}")
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise ValueError(f"无法添加向量到索引: {e}")

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
        embedding_util: EmbeddingUtil,
        data_path: str,
        config: Optional[StorageConfig] = None,
        user_prefs_handler=None,
    ):
        super().__init__(embedding_util, data_path)
        self.config = config or StorageConfig()
        self.user_prefs_handler = user_prefs_handler

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

    async def _ensure_collection_loaded(self, collection_name: str):
        """确保指定的集合已被加载"""
        if (
            self.collection_name != collection_name
            or not self.metadata_store
            or not self.index_store
        ):
            # 需要切换或首次加载集合
            if self.metadata_store:
                self.metadata_store.close()
            if self.index_store:
                self.index_store.close()

            # 加载新集合
            db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
            index_path = os.path.join(self.data_path, f"{collection_name}.faiss")

            self.metadata_store = SQLiteMetadataStore(db_path)
            self.index_store = FAISSIndexStore(index_path, self.config)

            # 总是需要初始化索引，无论文件是否存在
            # TODO: 优化索引初始化逻辑，支持动态维度检测
            try:
                dimension = self.embedding_util.get_dimensions(
                    collection_name, self.user_prefs_handler
                )
                if dimension is None or dimension <= 0:
                    logger.warning(f"无法获取有效的embedding维度，使用默认值1536")
                    dimension = 1536  # 默认维度

                self.index_store.initialize(dimension)
                logger.debug(f"索引初始化成功，维度: {dimension}")
            except Exception as e:
                logger.error(f"索引初始化失败: {e}")
                # 使用默认维度重试
                try:
                    self.index_store.initialize(1536)
                    logger.info("使用默认维度1536重新初始化索引成功")
                except Exception as retry_e:
                    logger.error(f"使用默认维度重新初始化索引也失败: {retry_e}")
                    raise ValueError(f"索引初始化失败: {retry_e}")

            self.collection_name = collection_name

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
            dimension = self.embedding_util.get_dimensions(
                collection_name, self.user_prefs_handler
            )
            self.index_store.initialize(dimension)

            self.collection_name = collection_name
            logger.info(f"成功创建集合: {collection_name}")

    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在（包括空集合）"""
        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        # 只要有数据库文件就认为集合存在（空集合是正常的）
        return os.path.exists(db_path)

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """添加文档到集合"""
        async with self._lock:
            if not await self.collection_exists(collection_name):
                await self.create_collection(collection_name)

            if not documents:
                return []

            # 确保集合已加载
            await self._ensure_collection_loaded(collection_name)

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
            logger.info(f"开始为 {len(texts)} 个文档生成embedding向量...")
            logger.debug(
                f"文档文本长度: {[len(text) for text in texts[:5]]}..."
            )  # 显示前5个文档的文本长度

            embeddings = await self.embedding_util.get_embeddings_async(
                texts, collection_name, self.user_prefs_handler
            )
            logger.info(
                f"Embedding生成完成，结果: {len([e for e in embeddings if e is not None])}/{len(embeddings)} 个有效"
            )

            # 过滤有效向量
            valid_docs = []
            valid_embeddings = []
            for i, (doc, embedding) in enumerate(zip(enhanced_docs, embeddings)):
                if embedding is not None:
                    doc.embedding = embedding
                    valid_docs.append(doc)
                    valid_embeddings.append(embedding)
                    logger.debug(f"文档 {i}: 有效embedding，维度: {len(embedding)}")
                else:
                    logger.warning(f"文档 {i}: embedding为空，跳过")

            logger.info(f"过滤后有效文档: {len(valid_docs)}/{len(enhanced_docs)}")

            if not valid_docs:
                logger.error("没有有效的向量可以添加 - 所有embedding都失败了")
                return []

            # 添加到FAISS索引
            vectors = np.array(valid_embeddings).astype("float32")
            faiss.normalize_L2(vectors)

            # TODO: 添加索引健康检查和自动修复机制
            # 确保索引已正确初始化
            if self.index_store.index is None:
                logger.error("FAISS索引为None，尝试重新初始化")
                try:
                    # 重新初始化索引
                    dimension = vectors.shape[1] if len(vectors) > 0 else 1536
                    self.index_store.initialize(dimension)
                    logger.info(f"重新初始化索引成功，维度: {dimension}")
                except Exception as e:
                    logger.error(f"重新初始化索引失败: {e}")
                    raise ValueError(f"索引初始化失败，无法添加向量: {e}")

            start_id = self.index_store.add_vectors(vectors)
            logger.info(f"向量添加成功，起始ID: {start_id}, 添加数量: {len(vectors)}")

            # 添加到SQLite - 修复ID映射问题
            doc_ids = []
            for i, doc in enumerate(valid_docs):
                vector_id = start_id + i
                doc.id = f"{collection_name}_{vector_id}"
                doc.vector_id = vector_id
                self.metadata_store.add_document(doc, vector_id)
                doc_ids.append(doc.id)

            logger.debug(
                f"已分配 {len(doc_ids)} 个文档ID (范围: {start_id}-{start_id + len(doc_ids) - 1})"
            )

            # 保存索引
            self.index_store.save_index()

            # 数据一致性验证
            await self._validate_data_consistency(collection_name, len(valid_docs))

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
        async with self._lock:
            logger.debug(
                f"开始搜索，集合: {collection_name}, 查询: '{query_text}', top_k: {top_k}"
            )

            if not await self.collection_exists(collection_name):
                logger.warning(f"集合 '{collection_name}' 不存在")
                return []

            # 确保集合已加载
            await self._ensure_collection_loaded(collection_name)
            logger.debug(f"集合 '{collection_name}' 已加载")

            # 检查集合是否有数据
            doc_count = await self.count_documents(collection_name)
            logger.info(f"集合 '{collection_name}' 包含 {doc_count} 个文档")

            if doc_count == 0:
                logger.info(f"集合 '{collection_name}' 为空，无法搜索")
                return []

            # 获取查询向量
            logger.debug("开始生成查询向量...")
            query_embedding = await self.embedding_util.get_embedding_async(
                query_text, collection_name, self.user_prefs_handler
            )
            if query_embedding is None:
                logger.error("无法生成查询向量")
                return []

            logger.debug(f"查询向量生成成功，维度: {len(query_embedding)}")

            # 向量搜索
            query_vector = np.array([query_embedding]).astype("float32")
            faiss.normalize_L2(query_vector)

            # TODO: 添加搜索前的索引状态检查
            # 确保索引可用于搜索
            if self.index_store.index is None:
                logger.error("FAISS索引为None，无法执行搜索")
                return []

            index_total = self.index_store.index.ntotal
            logger.info(f"索引包含 {index_total} 个向量")

            if index_total == 0:
                logger.info(f"集合 '{collection_name}' 的索引为空，没有可搜索的向量")
                return []

            logger.debug("开始执行向量搜索...")
            distances, indices = self.index_store.search_vectors(
                query_vector, top_k * 2
            )
            logger.debug(
                f"向量搜索完成，distances: {distances[0][: min(5, len(distances[0]))]}"
            )
            logger.debug(
                f"向量搜索完成，indices: {indices[0][: min(10, len(indices[0]))]}..."
            )

            # 获取文档 - 修复数据类型问题
            vector_ids = [
                int(idx) for idx in indices[0] if idx != -1
            ]  # 转换为Python int
            logger.debug(f"有效向量ID: {vector_ids[:10]}...")  # 只显示前10个
            logger.info(f"找到 {len(vector_ids)} 个有效向量ID")

            if not vector_ids:
                logger.warning("没有找到有效的向量ID")
                return []

            # 使用类的实例变量而不是创建新实例
            logger.debug("开始获取文档元数据...")
            docs = self.metadata_store.get_documents_by_vector_ids(vector_ids, filters)
            logger.info(f"获取到 {len(docs)} 个文档")

            # 构建搜索结果
            results = []
            doc_dict = {doc.vector_id: doc for doc in docs}  # 通过vector_id快速查找文档
            logger.debug(
                f"文档字典包含向量ID: {list(doc_dict.keys())[:10]}..."
            )  # 只显示前10个

            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx in doc_dict:  # 有效索引且文档存在（通过过滤器）
                    doc = doc_dict[idx]
                    similarity = 1.0 - (distance / 2.0)  # 转换为相似度
                    results.append(SearchResult(document=doc, score=float(similarity)))
                    logger.debug(
                        f"添加结果: vector_id={idx}, similarity={similarity:.4f}"
                    )

            # 按相似度排序
            results.sort(key=lambda x: x.score, reverse=True)
            final_results = results[:top_k]

            logger.info(f"搜索完成，返回 {len(final_results)} 个结果")
            for i, result in enumerate(final_results):
                logger.debug(
                    f"最终结果 {i + 1}: score={result.score:.4f}, doc_id={result.document.id}"
                )

            return final_results

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
                content, collection_name, self.user_prefs_handler
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
        try:
            # 删除文件（无论集合是否完整都尝试删除相关文件）
            db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
            index_path = os.path.join(self.data_path, f"{collection_name}.faiss")

            deleted_files = []
            if os.path.exists(db_path):
                os.remove(db_path)
                deleted_files.append("enhanced.db")
            if os.path.exists(index_path):
                os.remove(index_path)
                deleted_files.append("faiss")

            if deleted_files:
                logger.info(
                    f"成功删除集合 {collection_name} 的文件: {', '.join(deleted_files)}"
                )
                return True
            else:
                logger.warning(f"集合 {collection_name} 没有找到相关文件")
                return False
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

    async def cleanup_corrupted_collections(self) -> List[str]:
        """清理损坏的集合（数据库文件存在但无法正常访问的集合）"""
        corrupted_collections = []
        if not os.path.exists(self.data_path):
            return corrupted_collections

        for filename in os.listdir(self.data_path):
            if filename.endswith(".enhanced.db"):
                collection_name = filename[:-12]  # 移除.enhanced.db
                db_path = os.path.join(self.data_path, filename)

                # 检查数据库文件是否损坏
                try:
                    # 尝试简单的数据库连接测试
                    import sqlite3

                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    conn.close()
                except Exception as e:
                    # 如果数据库文件损坏，删除它
                    try:
                        os.remove(db_path)
                        # 同时删除可能存在的faiss文件
                        index_path = os.path.join(
                            self.data_path, f"{collection_name}.faiss"
                        )
                        if os.path.exists(index_path):
                            os.remove(index_path)
                        corrupted_collections.append(collection_name)
                        logger.info(f"已清理损坏的集合文件: {collection_name}")
                    except Exception as cleanup_error:
                        logger.error(
                            f"清理损坏集合失败 {collection_name}: {cleanup_error}"
                        )

        return corrupted_collections

    async def count_documents(self, collection_name: str) -> int:
        """统计文档数量"""
        if not await self.collection_exists(collection_name):
            return 0

        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        metadata_store = SQLiteMetadataStore(db_path)
        count = metadata_store.count_documents()
        metadata_store.close()
        return count

    @contextmanager
    def get_metadata_store_for_collection(self, collection_name: str):
        """获取指定集合的元数据存储的上下文管理器"""
        db_path = os.path.join(self.data_path, f"{collection_name}.enhanced.db")
        store = SQLiteMetadataStore(db_path)
        try:
            yield store
        finally:
            store.close()

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

    async def _validate_data_consistency(
        self, collection_name: str, expected_docs: int
    ):
        """验证数据一致性：确保FAISS索引和数据库同步"""
        try:
            # 获取索引中的向量数量
            index_count = self.index_store.count_vectors()

            # 获取数据库中的文档数量
            db_count = await self.count_documents(collection_name)

            logger.info(f"数据一致性检查:")
            logger.info(f"  FAISS索引向量数: {index_count}")
            logger.info(f"  数据库文档数: {db_count}")
            logger.info(f"  预期文档数: {expected_docs}")

            if index_count != db_count:
                logger.error(f"数据不一致！索引和数据库数量不匹配")
                logger.error(f"  索引: {index_count}, 数据库: {db_count}")

                # 检查具体的vector_id范围
                with self.get_metadata_store_for_collection(collection_name) as store:
                    all_docs = store.get_all_documents()
                    if all_docs:
                        vector_ids = [vid for _, vid in all_docs]
                        logger.info(
                            f"  数据库vector_id范围: {min(vector_ids)}-{max(vector_ids)}"
                        )
                        logger.info(f"  数据库vector_id总数: {len(set(vector_ids))}")

                        # 检查是否有缺失的ID
                        expected_range = set(range(index_count))
                        actual_range = set(vector_ids)
                        missing_ids = expected_range - actual_range
                        extra_ids = actual_range - expected_range

                        if missing_ids:
                            logger.error(
                                f"  缺失的vector_id: {sorted(list(missing_ids))[:10]}..."
                            )
                        if extra_ids:
                            logger.error(
                                f"  多余的vector_id: {sorted(list(extra_ids))[:10]}..."
                            )
                    else:
                        logger.error("  数据库中没有文档记录")
            else:
                logger.info("✅ 数据一致性检查通过")

        except Exception as e:
            logger.error(f"数据一致性验证失败: {e}")
