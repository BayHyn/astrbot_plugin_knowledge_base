from typing import List, Dict
import asyncio

# 导入semchunk分块库
try:
    import semchunk
    SEMCHUNK_AVAILABLE = True
except ImportError:
    SEMCHUNK_AVAILABLE = False
    raise ImportError("semchunk库是必需的，请安装: pip install semchunk")


class TextSplitterUtil:
    """
    文本分割器，使用semchunk进行语义化分块
    
    注意：semchunk原本是基于token进行分块的，但这里通过自定义token_counter
    将其转换为基于字符数的分块，以保持与原有接口的一致性。
    
    semchunk的优势：
    - 语义化分块：在句子、段落等语义边界分割
    - 自动保护Markdown结构（代码块、表格、链接等）
    - 高效的重叠处理
    - 优化的内存使用
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        use_optimized: bool = True,  # 保持兼容性参数
        preserve_code_blocks: bool = True,  # 保持兼容性参数
        preserve_tables: bool = True,  # 保持兼容性参数
    ):
        """
        初始化文本分割器。
        Args:
            chunk_size: 每个块的目标大小 (字符数)
            chunk_overlap: 块之间的重叠大小
            use_optimized: 兼容性参数，已弃用
            preserve_code_blocks: 兼容性参数，已弃用
            preserve_tables: 兼容性参数，已弃用
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 创建字符计数器：确保semchunk按字符数分块
        self.token_counter = self._character_counter
        
        # 计算重叠比例（semchunk支持相对重叠和绝对重叠）
        if chunk_overlap >= 1:
            # 绝对重叠（字符数量）
            self.overlap = min(chunk_overlap, chunk_size - 1)
        else:
            # 相对重叠（比例）
            self.overlap = min(chunk_overlap / chunk_size, 0.5) if chunk_size > 0 else 0

    def _character_counter(self, text: str) -> int:
        """
        字符计数器，确保semchunk按字符数进行分块
        Args:
            text: 要计数的文本
        Returns:
            字符数量
        """
        return len(text)

    async def split_text(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> List[str]:
        """
        异步将文本分割成块。
        Args:
            text: 待分割的文本
            chunk_size: 覆盖默认的块大小
            overlap: 覆盖默认的重叠大小
        Returns:
            分割后的文本块列表
        """
        if not text or not text.strip():
            return []

        # 使用提供的参数或默认参数
        actual_chunk_size = chunk_size or self.chunk_size
        actual_overlap = overlap or self.chunk_overlap
        
        # 计算重叠比例
        if actual_overlap >= 1:
            # 绝对重叠
            overlap_param = min(actual_overlap, actual_chunk_size - 1)
        else:
            # 相对重叠
            overlap_param = min(actual_overlap / actual_chunk_size, 0.5) if actual_chunk_size > 0 else 0

        try:
            # 在执行器中运行semchunk，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                self._sync_split_text,
                text,
                actual_chunk_size,
                overlap_param
            )
            return chunks

        except Exception as e:
            # 如果semchunk完全失败，抛出异常而不是降级
            raise ValueError(f"semchunk分块失败: {e}")

    def _sync_split_text(self, text: str, chunk_size: int, overlap_param: float) -> List[str]:
        """在线程池中执行的同步分块方法"""
        return semchunk.chunk(
            text=text,
            chunk_size=chunk_size,
            token_counter=self.token_counter,
            memoize=True,
            offsets=False,
            overlap=overlap_param if overlap_param > 0 else None,
            cache_maxsize=1000  # 限制缓存大小
        )

    # 保持同步版本的兼容性
    def split_text_sync(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> List[str]:
        """
        同步版本的文本分割（兼容性方法）
        Args:
            text: 待分割的文本
            chunk_size: 覆盖默认的块大小
            overlap: 覆盖默认的重叠大小
        Returns:
            分割后的文本块列表
        """
        if not text or not text.strip():
            return []

        actual_chunk_size = chunk_size or self.chunk_size
        actual_overlap = overlap or self.chunk_overlap
        
        if actual_overlap >= 1:
            overlap_param = min(actual_overlap, actual_chunk_size - 1)
        else:
            overlap_param = min(actual_overlap / actual_chunk_size, 0.5) if actual_chunk_size > 0 else 0

        try:
            return self._sync_split_text(text, actual_chunk_size, overlap_param)
        except Exception as e:
            raise ValueError(f"semchunk分块失败: {e}")

    async def get_chunk_metadata(self, text: str) -> List[Dict]:
        """获取文本块的详细元数据"""
        chunks = await self.split_text(text)

        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata_dict = {
                "chunk_index": i,
                "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "length": len(chunk),
                "splitter_type": "semchunk",
            }
            metadata_list.append(metadata_dict)

        return metadata_list

    def get_splitter_info(self) -> Dict[str, any]:
        """获取当前使用的分割器信息"""
        return {
            "splitter_type": "semchunk",
            "semchunk_available": SEMCHUNK_AVAILABLE,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "overlap_param": self.overlap,
        }