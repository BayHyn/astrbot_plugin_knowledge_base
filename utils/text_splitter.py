from typing import List, Dict

# 导入semchunk分块库
try:
    import semchunk
    SEMCHUNK_AVAILABLE = True
except ImportError:
    SEMCHUNK_AVAILABLE = False
    raise ImportError("semchunk库是必需的，请安装: pip install semchunk")


class TextSplitterUtil:
    """文本分割器，使用semchunk进行语义化分块"""

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
        
        # 创建字符计数器（semchunk使用token计数，这里用字符长度模拟）
        self.token_counter = len
        
        # 计算重叠比例（semchunk支持相对重叠和绝对重叠）
        if chunk_overlap >= 1:
            # 绝对重叠（token数量）
            self.overlap = min(chunk_overlap, chunk_size - 1)
        else:
            # 相对重叠（比例）
            self.overlap = min(chunk_overlap / chunk_size, 0.5) if chunk_size > 0 else 0

    def split_text(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> List[str]:
        """
        将文本分割成块。
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
            # 使用semchunk进行分块
            chunks = semchunk.chunk(
                text=text,
                chunk_size=actual_chunk_size,
                token_counter=self.token_counter,
                memoize=True,
                offsets=False,
                overlap=overlap_param if overlap_param > 0 else None,
                cache_maxsize=1000  # 限制缓存大小
            )
            return chunks

        except Exception as e:
            # 如果semchunk完全失败，抛出异常而不是降级
            raise ValueError(f"semchunk分块失败: {e}")

    def get_chunk_metadata(self, text: str) -> List[Dict]:
        """获取文本块的详细元数据"""
        chunks = self.split_text(text)

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
