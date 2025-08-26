import os
import asyncio
import json
import gc
from typing import List, Dict, Tuple, Set, Optional

# 引入 cachetools 和 Lock, Set, Optional
try:
    from cachetools import LRUCache
except ImportError:
    raise ImportError("Please install cachetools: pip install cachetools")

from .base import (
    VectorDBBase,
    Document,
    ProcessingBatch,
    DEFAULT_BATCH_SIZE,
)
from astrbot.api import logger
from astrbot.core.db.vec_db.faiss_impl import FaissVecDB
from astrbot.core.provider.provider import EmbeddingProvider
from ..utils.embedding import EmbeddingSolutionHelper
from .faiss_store import FaissStore as OldFaissStore

# 定义默认的缓存大小
DEFAULT_MAX_CACHE_SIZE = 3
# 定义内存管理相关常量
DEFAULT_MEMORY_BATCH_SIZE = 50  # 内存中同时处理的文档数量
MAX_DOCUMENTS_WARNING_THRESHOLD = 5000  # 大文件警告阈值


def _check_pickle_file(file_path: str) -> bool:
    """检查文件是否为 Pickle 格式"""
    try:
        if not os.path.exists(file_path):
            return False
        with open(file_path, "rb") as f:
            magic = f.read(2)
            # 兼容python3.8之前的pickle协议版本
            return magic in [b"\x80\x04", b"\x80\x03", b"\x80\x02"]
    except Exception:
        return False


class AstrBotEmbeddingProviderWrapper(EmbeddingProvider):
    """AstrBot Embedding Provider 包装类"""

    def __init__(
        self,
        embedding_util: EmbeddingSolutionHelper,
        collection_name: str,
    ):
        self.embedding_util = embedding_util
        self.collection_name = collection_name

    async def get_embedding(self, text: str) -> List[float]:
        vec = await self.embedding_util.get_embedding_async(text, self.collection_name)
        if not vec:
            raise ValueError(
                "获取向量失败，返回的向量为空或无效。请检查输入文本和配置。"
            )
        return vec

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的嵌入"""
        vecs = await self.embedding_util.get_embeddings_async(
            texts, self.collection_name
        )
        if not vecs:
            raise ValueError(
                "获取向量失败，返回的向量为空或无效。请检查输入文本和配置。"
            )
        return vecs

    def get_dim(self) -> int:
        return self.embedding_util.get_dimensions(self.collection_name)


class FaissStore(VectorDBBase):
    """
    对 AstrBot FaissVecDB 的包装类，以适应 KB 的接口规范
    使用 LRU Cache 按需加载和管理知识库集合
    """

    def __init__(
        self,
        embedding_util: EmbeddingSolutionHelper,
        data_path: str,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ):
        super().__init__(embedding_util, data_path)
        # self.vecdbs: Dict[str, FaissVecDB] = {} # 被 cache 替代

        # ---- LRU Cache 相关 ----
        # LRU 缓存，存储 collection_name -> FaissVecDB 实例
        self.cache: LRUCache[str, FaissVecDB] = LRUCache(maxsize=max_cache_size)
        # 记录磁盘上所有已知的新格式集合名称（无论是否加载）
        self._all_known_collections: Set[str] = set()
        # 加载锁，防止同一集合并发加载
        self._locks: Dict[str, asyncio.Lock] = {}
        self.max_cache_size = max_cache_size
        # 内存管理批次大小
        self.memory_batch_size = DEFAULT_MEMORY_BATCH_SIZE
        logger.info(
            f"[知识库-缓存] FaissStore LRU缓存初始化完成: 最大缓存大小={max_cache_size}, 内存批次大小={self.memory_batch_size}"
        )
        # ------------------------

        self._old_faiss_store: Optional[OldFaissStore] = None
        self._old_collections: Dict[str, str] = {}  # 记录所有旧格式的集合
        self.embedding_utils: Dict[str, AstrBotEmbeddingProviderWrapper] = {}
        os.makedirs(self.data_path, exist_ok=True)

    async def initialize(self):
        """初始化：仅扫描磁盘，不加载任何集合到内存"""
        logger.info(f"[知识库-初始化] 开始扫描Faiss存储路径: {self.data_path}")
        # 初始化时只扫描，不加载
        await self._scan_collections_on_disk()
        logger.info(
            f"[知识库-初始化] 扫描完成 - 新格式集合: {len(self._all_known_collections)}个 {list(self._all_known_collections)}, "
            f"旧格式集合: {len(self._old_collections)}个 {list(self._old_collections.keys())}"
        )

    def _get_collection_meta(self, collection_name: str) -> Tuple[str, str, str, str]:
        """工具函数：根据集合名获取真实名称, file_id 和路径"""
        true_coll_name = (
            self.embedding_util.user_prefs_handler.get_collection_name_by_file_id(
                collection_name
            )
        )
        # 检查元数据
        collection_md = (
            self.embedding_util.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            ).get(collection_name, {})
        )

        if true_coll_name:
            # collection_name is actually a file_id
            file_id = collection_name
            final_collection_name = true_coll_name
        elif collection_md:
            # collection_name is a true name, get file_id from metadata
            file_id = collection_md.get("file_id", collection_name)
            final_collection_name = collection_name
        else:
            # fallback
            file_id = collection_name
            final_collection_name = collection_name

        index_path = os.path.join(self.data_path, f"{file_id}.index")
        storage_path = os.path.join(self.data_path, f"{file_id}.db")
        _old_storage_path = os.path.join(self.data_path, f"{file_id}.docs")
        return (
            final_collection_name,
            file_id,
            index_path,
            storage_path,
            _old_storage_path,
        )

    async def _scan_collections_on_disk(self):
        """扫描磁盘目录，识别新旧集合，填充 _all_known_collections 和 _old_collections"""
        self._all_known_collections.clear()
        self._old_collections.clear()
        if not os.path.exists(self.data_path):
            return

        scanned_file_ids = set()
        # 优先处理 .index 和 .db 文件
        all_files = os.listdir(self.data_path)
        relevant_extensions = (".index", ".db", ".docs")

        for filename in all_files:
            if not filename.endswith(relevant_extensions):
                continue

            base, ext = os.path.splitext(filename)
            if base in scanned_file_ids:
                continue

            file_id = base
            collection_name, _, index_path, storage_path, _old_storage_path = (
                self._get_collection_meta(file_id)
            )

            is_old = False
            # 检查是否为旧格式
            if _check_pickle_file(storage_path) or os.path.exists(_old_storage_path):
                is_old = True
            # 如果 .index 和 .db 都存在，认为是新格式 (除非 .db 是pickle 或存在 .docs)
            elif os.path.exists(index_path) and os.path.exists(storage_path):
                is_old = False
            # 如果只有 .docs，认为是旧格式
            elif ext == ".docs" and not os.path.exists(index_path):
                is_old = True
            else:
                # 其他情况，例如只有 .index 或只有 .db (非pickle)，暂时跳过或认为是新格式不完整
                # 为简单起见，如果存在 index 和 db 之一且非旧格式，就认为是新格式
                if ext in (".index", ".db"):
                    is_old = False
                else:
                    continue  # 忽略不明确的文件

            scanned_file_ids.add(file_id)
            if is_old:
                self._old_collections[collection_name] = collection_name
                logger.debug(f"[知识库-扫描] 发现旧格式集合: {collection_name} (file_id: {file_id})")
            else:
                self._all_known_collections.add(collection_name)
                logger.debug(f"[知识库-扫描] 发现新格式集合: {collection_name} (file_id: {file_id})")

        # 如果发现了旧集合，初始化旧存储实例
        if self._old_collections and not self._old_faiss_store:
            logger.info(f"[知识库-扫描] 检测到旧格式集合，初始化OldFaissStore处理器...")
            self._old_faiss_store = OldFaissStore(self.embedding_util, self.data_path)
            await self._old_faiss_store.initialize()

    async def _perform_load(
        self, collection_name: str, index_path: str, storage_path: str
    ) -> FaissVecDB:
        """执行实际的加载/创建 FaissVecDB 逻辑，不涉及缓存和锁"""
        logger.info(f"[知识库-加载] 开始加载/创建Faiss集合实例: '{collection_name}'")
        self.embedding_utils[collection_name] = AstrBotEmbeddingProviderWrapper(
            embedding_util=self.embedding_util,
            collection_name=collection_name,
        )
        params = {
            "doc_store_path": storage_path,
            "index_store_path": index_path,
            "embedding_provider": self.embedding_utils[collection_name],
        }
        rerank_prov = self.embedding_util.get_rerank_provider(collection_name)
        if rerank_prov:
            params["rerank_provider"] = rerank_prov
        vecdb = FaissVecDB(**params)
        await vecdb.initialize()
        logger.info(f"[知识库-加载] Faiss集合实例 '{collection_name}' 加载/创建完成")
        return vecdb

    async def _evict_lru_if_needed(self):
        """如果缓存已满，则移除并关闭最少使用的集合"""
        evicted_count = 0
        while len(self.cache) >= self.max_cache_size and self.max_cache_size > 0:
            try:
                lru_key, lru_vecdb = self.cache.popitem()
                logger.info(
                    f"[知识库-缓存] 缓存已满(max={self.max_cache_size})，移出最少使用的集合: '{lru_key}'"
                )
                self.embedding_utils.pop(lru_key, None)
                self._locks.pop(lru_key, None)  # 清理锁
                try:
                    await lru_vecdb.close()
                    logger.info(f"[知识库-缓存] 成功关闭被移出的集合: '{lru_key}'")
                    evicted_count += 1
                except Exception as close_e:
                    logger.error(f"[知识库-缓存] 关闭被移出的集合 '{lru_key}' 时发生错误: {close_e}")
            except KeyError:
                # 缓存为空
                break
            except Exception as e:
                logger.error(f"[知识库-缓存] 缓存移出过程发生未知错误: {e}")
                break

        # 如果有移出操作，触发垃圾回收
        if evicted_count > 0:
            gc.collect()
            logger.debug(f"已移出 {evicted_count} 个集合，触发垃圾回收")

    async def _unload_collection(self, collection_name: str):
        """从缓存中卸载并关闭一个指定的集合"""
        vecdb_to_close = self.cache.pop(collection_name, None)
        self.embedding_utils.pop(collection_name, None)
        self._locks.pop(collection_name, None)  # 清理锁
        if vecdb_to_close:
            logger.info(f"从缓存中卸载并关闭集合: '{collection_name}'")
            try:
                await vecdb_to_close.close()
            except Exception as e:
                logger.error(f"关闭集合 '{collection_name}' 时出错: {e}")

    async def _get_or_load_vecdb(
        self, collection_name: str, for_create: bool = False
    ) -> Optional[FaissVecDB]:
        """
        核心函数：从缓存获取或按需加载集合
        1. 检查缓存
        2. 缓存未命中则加锁
        3. 锁内再次检查缓存（Double-Check Locking）
        4. 检查是否需要移出 LRU
        5. 加载集合
        6. 放入缓存
        """
        # 1. 旧集合或已在缓存中，直接返回
        if collection_name in self._old_collections:
            return None
        if collection_name in self.cache:
            # 访问即更新其在 LRU 中的位置
            return self.cache[collection_name]

        # 2. 获取或创建针对此集合的锁
        lock = self._locks.setdefault(collection_name, asyncio.Lock())

        async with lock:
            # 3. 锁内再次检查，防止在等待锁期间其他协程已加载
            if collection_name in self.cache:
                return self.cache[collection_name]

            logger.info(f"[知识库-加载] 缓存未命中，准备加载集合: '{collection_name}'")

            _, _, index_path, storage_path, _ = self._get_collection_meta(
                collection_name
            )

            # 如果不是创建操作，且文件不存在，则不加载
            if not for_create and not (
                os.path.exists(index_path) and os.path.exists(storage_path)
            ):
                logger.warning(f"[知识库-加载] 警告: 集合 '{collection_name}' 的文件不存在，无法加载。索引文件: {index_path}, 存储文件: {storage_path}")
                # self._locks.pop(collection_name, None) # 加载失败，清理锁
                return None

            # 4. 加载前检查并执行移出操作
            await self._evict_lru_if_needed()

            # 5. 执行加载
            try:
                vecdb = await self._perform_load(
                    collection_name, index_path, storage_path
                )
                # 6. 放入缓存
                self.cache[collection_name] = vecdb
                self._all_known_collections.add(collection_name)  # 确保已记录
                logger.info(
                    f"[知识库-加载] 集合 '{collection_name}' 已加载并放入缓存。当前缓存大小: {len(self.cache)}/{self.max_cache_size}"
                )
                return vecdb
            except Exception as e:
                logger.error(f"[知识库-加载] 加载知识库集合(FAISS) '{collection_name}' 时出错: {type(e).__name__} - {str(e)}")
                # 清理可能残留的状态
                self.cache.pop(collection_name, None)
                self.embedding_utils.pop(collection_name, None)
                # self._locks.pop(collection_name, None) # 加载失败，清理锁
                return None
        # 锁自动释放

    # async def _load_collection(self, collection_name: str): # 废弃
    # async def _load_all_collections(self): # 废弃，由 _scan_collections_on_disk 替代扫描功能

    async def create_collection(self, collection_name: str):
        """创建并加载一个新集合到缓存"""
        if await self.collection_exists(collection_name):
            # 如果已存在（在磁盘或旧存储中），尝试加载到缓存（如果还不在）
            logger.info(f"[知识库-创建] Faiss集合 '{collection_name}' 已存在，尝试加载到缓存")
            await self._get_or_load_vecdb(collection_name)
            return

        logger.info(f"[知识库-创建] 开始创建新Faiss集合 '{collection_name}'")
        # 保存偏好设置
        await self.embedding_util.user_prefs_handler.save_user_preferences()

        # 使用 _get_or_load_vecdb 进行创建，它会处理锁、缓存移出和加载
        # 设置 for_create=True 使得即使文件不存在也会继续 _perform_load
        vecdb = await self._get_or_load_vecdb(collection_name, for_create=True)

        if vecdb:
            # 新创建的集合需要显式保存一下索引文件
            await vecdb.embedding_storage.save_index()
            # _get_or_load_vecdb 已经将其加入 _all_known_collections
            logger.info(f"[知识库-创建] Faiss集合 '{collection_name}' 创建成功并已加载到缓存")
        else:
            logger.error(f"[知识库-创建] Faiss集合 '{collection_name}' 创建或加载失败")

    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在于磁盘（新格式）或旧存储中"""
        # 检查已知的（扫描到的或创建的）新格式集合，以及旧格式集合
        return (
            collection_name in self._all_known_collections
            or collection_name in self._old_collections
        )

    async def _process_documents_batch(
        self,
        documents: List[Document],
        collection_name: str,
        vecdb: FaissVecDB,
    ) -> Tuple[List[str], int]:
        """
        顺序处理一批文档，优化内存使用
        返回: (成功添加的文档ID列表, 失败数量)
        """
        if not vecdb:
            logger.error(f"[知识库-批次处理] 致命错误: 集合 '{collection_name}' 的 vecdb 实例为空，无法处理文档")
            return [], len(documents)

        doc_ids = []
        failed_count = 0
        batch_size = len(documents)
        
        logger.debug(f"[知识库-批次处理] 开始逐个处理 {batch_size} 个文档，集合: '{collection_name}'")
        
        for i, doc in enumerate(documents):
            try:
                # 获取文档预览用于日志
                doc_preview = doc.text_content[:50].replace("\n", " ") if doc.text_content else "[空文档]"
                
                doc_id = await vecdb.insert(
                    content=doc.text_content,
                    metadata=doc.metadata,
                )
                doc_ids.append(doc_id)
                
                # 详细的进度日志（每10个文档记录一次）
                if (i + 1) % 10 == 0 or (i + 1) == batch_size:
                    logger.debug(f"[知识库-批次处理] 进度: {i+1}/{batch_size} 个文档已处理，最新: '{doc_preview}...'")
                
                # 及时清理文档引用以释放内存
                doc.text_content = ""
                doc.metadata.clear()
                
            except Exception as e:
                failed_count += 1
                excerpt = doc.text_content[:50].replace("\n", " ") if doc.text_content else "[空文档]"
                logger.error(
                    f"[知识库-批次处理] 文档处理失败: 第{i+1}/{batch_size}个文档 '{excerpt}...' 添加到集合 '{collection_name}' 失败，"
                    f"错误类型: {type(e).__name__}，错误详情: {str(e)}"
                )
            
            # 每处理一定数量的文档后触发垃圾回收
            if (i + 1) % 20 == 0:
                gc.collect()
                logger.debug(f"[知识库-批次处理] 已处理 {i+1} 个文档，执行内存垃圾回收")
                
        # 清空整个批次的文档列表
        documents.clear()
        
        if failed_count == 0:
            logger.debug(f"[知识库-批次处理] ✅ 批次处理完成: 集合 '{collection_name}' 成功处理 {len(doc_ids)} 个文档")
        else:
            logger.warning(f"[知识库-批次处理] ⚠️ 批次处理完成: 集合 '{collection_name}' 成功 {len(doc_ids)} 个，失败 {failed_count} 个文档")
            
        return doc_ids, failed_count

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """
        向指定集合中添加文档，使用顺序处理优化内存使用
        """
        # 处理旧集合
        if collection_name in self._old_collections:
            if self._old_faiss_store:
                logger.info(f"[知识库-添加文档] 检测到旧格式集合 '{collection_name}'，使用旧存储引擎处理")
                return await self._old_faiss_store.add_documents(
                    collection_name, documents
                )
            else:
                logger.error(f"[知识库-添加文档] 致命错误: 旧集合 '{collection_name}' 存在但 OldFaissStore 未初始化，请检查插件配置")
                return []

        # 检查或创建集合
        if not await self.collection_exists(collection_name):
            logger.info(f"[知识库-添加文档] 目标集合 '{collection_name}' 不存在，开始自动创建新集合")
            await self.create_collection(collection_name)
        else:
            logger.info(f"[知识库-添加文档] 目标集合 '{collection_name}' 已存在，准备添加文档")

        # 获取集合实例
        vecdb = await self._get_or_load_vecdb(collection_name)
        if not vecdb:
            logger.error(f"[知识库-添加文档] 严重错误: 无法获取或加载集合 '{collection_name}' 的向量数据库实例，文档添加操作失败")
            return []

        total_documents = len(documents)
        if total_documents == 0:
            logger.warning(f"[知识库-添加文档] 警告: 传入的文档列表为空，集合 '{collection_name}' 无需处理")
            return []

        # 内存使用警告和系统状态检查
        if total_documents > MAX_DOCUMENTS_WARNING_THRESHOLD:
            logger.warning(
                f"[知识库-添加文档] 内存警告: 准备一次性处理 {total_documents} 个文档 (超过阈值 {MAX_DOCUMENTS_WARNING_THRESHOLD})，"
                f"建议分批上传以避免内存溢出。当前内存批次大小: {self.memory_batch_size}"
            )

        logger.info(f"[知识库-添加文档] 开始处理: 集合='{collection_name}', 总文档数={total_documents}, 批次大小={self.memory_batch_size}")
        
        all_doc_ids = []
        total_failed = 0
        processed_count = 0
        total_batches = (total_documents + self.memory_batch_size - 1) // self.memory_batch_size

        # 分批顺序处理，避免内存激增
        for i in range(0, total_documents, self.memory_batch_size):
            batch_end = min(i + self.memory_batch_size, total_documents)
            batch_documents = documents[i:batch_end]
            batch_size = len(batch_documents)
            current_batch = i // self.memory_batch_size + 1
            
            logger.info(f"[知识库-添加文档] 处理批次 {current_batch}/{total_batches}: 文档范围 {i+1}-{batch_end} (共{batch_size}个)")
            
            try:
                batch_doc_ids, batch_failed = await self._process_documents_batch(
                    batch_documents, collection_name, vecdb
                )
                
                all_doc_ids.extend(batch_doc_ids)
                total_failed += batch_failed
                processed_count += batch_size
                
                success_in_batch = len(batch_doc_ids)
                logger.info(f"[知识库-添加文档] 批次 {current_batch} 完成: 成功 {success_in_batch}/{batch_size}" + 
                           (f", 失败 {batch_failed}" if batch_failed > 0 else ""))
                
                # 每处理几个批次后进行垃圾回收
                if current_batch % 3 == 0:
                    gc.collect()
                    logger.debug(f"[知识库-添加文档] 批次 {current_batch} 后执行内存垃圾回收")
                    
            except Exception as e:
                logger.error(f"[知识库-添加文档] 批次处理异常: 批次 {current_batch}/{total_batches} 发生严重错误，"
                           f"影响文档数量 {batch_size}，错误详情: {str(e)}")
                total_failed += batch_size

        # 保存索引
        try:
            logger.info(f"[知识库-添加文档] 开始保存集合 '{collection_name}' 的索引文件...")
            await vecdb.embedding_storage.save_index()
            logger.info(f"[知识库-添加文档] 集合 '{collection_name}' 索引文件保存成功")
        except Exception as e:
            logger.error(f"[知识库-添加文档] 索引保存失败: 集合 '{collection_name}' 索引文件保存时发生错误，"
                        f"可能影响后续搜索功能，错误详情: {str(e)}")

        # 最终清理和统计
        documents.clear()
        gc.collect()
        
        success_count = len(all_doc_ids)
        success_rate = (success_count / total_documents * 100) if total_documents > 0 else 0
        
        if total_failed == 0:
            logger.info(f"[知识库-添加文档] ✅ 任务完成: 集合 '{collection_name}' 成功添加 {success_count}/{total_documents} 个文档 (100%)")
        else:
            logger.warning(f"[知识库-添加文档] ⚠️ 任务完成(有失败): 集合 '{collection_name}' 成功 {success_count}/{total_documents} 个文档 "
                          f"({success_rate:.1f}%), 失败 {total_failed} 个，请检查上方错误日志")
        
        return all_doc_ids

    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        logger.info(f"[知识库-搜索] 开始搜索: 集合='{collection_name}', 查询文本预览='{query_text[:30]}...', top_k={top_k}")
        
        if not await self.collection_exists(collection_name):
            logger.warning(f"[知识库-搜索] 警告: Faiss集合 '{collection_name}' 不存在，搜索结果为空")
            return []

        # 首先处理旧集合
        if collection_name in self._old_collections:
            if self._old_faiss_store:
                logger.info(f"[知识库-搜索] 使用旧存储引擎处理集合 '{collection_name}' 的搜索")
                return await self._old_faiss_store.search(
                    collection_name, query_text, top_k
                )
            else:
                logger.error(
                    f"[知识库-搜索] 错误: 旧集合 '{collection_name}' 存在但 OldFaissStore 未初始化"
                )
                return []

        # 获取或加载集合实例
        logger.debug(f"[知识库-搜索] 获取集合 '{collection_name}' 的向量数据库实例")
        vecdb = await self._get_or_load_vecdb(collection_name)
        if not vecdb:
            logger.error(f"[知识库-搜索] 错误: 无法获取或加载集合 '{collection_name}' 的向量数据库实例，搜索失败")
            return []

        try:
            # 安全地检查是否使用重排序，记录调试信息
            has_rerank_attr = hasattr(vecdb, 'rerank_provider')
            if has_rerank_attr:
                rerank_provider_value = getattr(vecdb, 'rerank_provider', None)
                use_rerank = rerank_provider_value is not None
                logger.debug(f"[知识库-搜索] FaissVecDB有rerank_provider属性，值: {rerank_provider_value is not None}")
            else:
                use_rerank = False
                logger.debug(f"[知识库-搜索] FaissVecDB没有rerank_provider属性，使用普通搜索模式")
                
            if use_rerank:
                logger.debug(f"[知识库-搜索] 使用重排序模式搜索，初始检索数量: {max(20, top_k)}")
                results = await vecdb.retrieve(
                    query=query_text,
                    k=max(20, top_k),
                    rerank=True,
                )
                results = results[:top_k]
            else:
                logger.debug(f"[知识库-搜索] 使用普通模式搜索，检索数量: {top_k}")
                results = await vecdb.retrieve(query=query_text, k=top_k, rerank=False)
                
        except Exception as e:
            logger.error(f"[知识库-搜索] 搜索异常: 在集合 '{collection_name}' 中搜索时发生错误，"
                        f"错误类型: {type(e).__name__}，错误详情: {str(e)}")
            return []

        # 处理搜索结果
        ret = []
        failed_parse_count = 0
        for i, result in enumerate(results):
            if result is not None:
                try:
                    metadata = json.loads(result.data.get("metadata", "{}"))
                except json.JSONDecodeError as json_e:
                    failed_parse_count += 1
                    metadata = {}
                    logger.warning(
                        f"[知识库-搜索] JSON解析失败: 集合 {collection_name} 文档 {result.data.get('doc_id')} 元数据解析失败，错误: {str(json_e)}"
                    )
                doc = Document(
                    id=result.data.get("doc_id"),
                    embedding=[],  # 原始代码这里就是空
                    text_content=result.data.get("text", ""),
                    metadata=metadata,
                )
                ret.append((doc, result.similarity))
                
        # 详细的搜索结果日志
        if len(ret) == 0:
            logger.warning(f"[知识库-搜索] 搜索结果为空: 集合 '{collection_name}' 中未找到与查询 '{query_text[:30]}...' 相关的内容")
        else:
            avg_similarity = sum(score for _, score in ret) / len(ret)
            logger.info(
                f"[知识库-搜索] ✓ 搜索完成: 集合='{collection_name}', 查询='{query_text[:30]}...', "
                f"返回结果数={len(ret)}, 平均相似度={avg_similarity:.3f}"
                + (f", JSON解析失败={failed_parse_count}个" if failed_parse_count > 0 else "")
            )
        return ret

    async def delete_collection(self, collection_name: str) -> bool:
        if not await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 不存在，无需删除。")
            return False

        # 首先处理旧集合
        if collection_name in self._old_collections:
            self._old_collections.pop(collection_name, None)
            if self._old_faiss_store:
                return await self._old_faiss_store.delete_collection(collection_name)
            return False

        # 如果集合在缓存中，先卸载并关闭它
        await self._unload_collection(collection_name)
        # 从已知集合列表中移除
        self._all_known_collections.discard(collection_name)

        # 保持文件删除在线程中执行
        def _delete_sync():
            # self.vecdbs.pop(collection_name, None) # 改为 _unload_collection
            _, file_id, index_path, storage_path, _ = self._get_collection_meta(
                collection_name
            )

            try:
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(storage_path):
                    os.remove(storage_path)
                logger.info(
                    f"Faiss 集合文件 '{collection_name}' (file_id: {file_id}) 已删除。"
                )
                return True
            except Exception as e:
                logger.error(f"删除 Faiss 集合 '{collection_name}' 文件时出错: {e}")
                return False

        return await asyncio.to_thread(_delete_sync)

    async def list_collections(self) -> List[str]:
        """列出所有已知的集合（包括缓存中的、磁盘上未加载的、旧格式的）"""
        # 重新扫描可能更准确，但为了效率，依赖初始化扫描和创建/删除时的维护
        # await self._scan_collections_on_disk()
        return list(self._all_known_collections) + list(self._old_collections.keys())

    async def count_documents(self, collection_name: str) -> int:
        if not await self.collection_exists(collection_name):
            return 0
        # 首先处理旧集合
        if collection_name in self._old_collections:
            if self._old_faiss_store:
                return await self._old_faiss_store.count_documents(collection_name)
            else:
                return 0

        # 获取或加载集合实例
        vecdb = await self._get_or_load_vecdb(collection_name)
        if not vecdb:
            logger.warning(f"无法获取或加载集合 '{collection_name}' 来计数。")
            return 0
        try:
            cnt = await vecdb.count_documents()
            return cnt
        except Exception as e:
            logger.error(f"获取集合 '{collection_name}' 文档数量时出错: {e}")
            return 0

    async def close(self):
        """关闭所有缓存中的集合和旧存储"""
        logger.info(f"正在关闭所有已加载的 Faiss 集合 (缓存大小: {len(self.cache)})...")
        # 复制 key 列表，因为 _unload_collection 会修改 self.cache
        try:
            collections_to_unload = list(self.cache.keys())
            for collection_name in collections_to_unload:
                await self._unload_collection(collection_name)

            self.cache.clear()
            self.embedding_utils.clear()
            self._locks.clear()
            self._all_known_collections.clear()
            logger.info("所有缓存中的 Faiss 集合已关闭和清理。")

            if self._old_faiss_store:
                logger.info("正在关闭 OldFaissStore...")
                await self._old_faiss_store.close()
                self._old_faiss_store = None
                self._old_collections.clear()
                logger.info("OldFaissStore 已关闭。")

            # 强制垃圾回收
            gc.collect()
            logger.debug("已触发垃圾回收释放内存")

        except Exception as e:
            logger.error(f"关闭 Faiss 集合时发生错误: {e}")
        logger.info("FaissStore 关闭完成。")
