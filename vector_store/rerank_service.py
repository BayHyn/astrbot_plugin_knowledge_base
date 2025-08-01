"""
重排序服务实现
支持API服务和多种重排序策略
"""

import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from astrbot.api import logger

from .base import Document, SearchResult
from .api_rerank_config import APIRerankConfig
from .api_rerank_service import APIRerankService


class SimpleReranker:
    """简单的重排序实现（基于规则）"""

    def __init__(self):
        self.weights = {"vector_score": 0.5, "keyword_score": 0.3, "recency_score": 0.2}

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]] | List[SearchResult],
        keyword_scores: Optional[List[Tuple[str, float]]] = None,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]] | List[SearchResult]:
        """基于规则的重排序"""
        if not documents:
            return []

        # 检查输入类型
        is_search_result = (
            isinstance(documents[0], SearchResult) if documents else False
        )

        # 构建关键词分数映射
        keyword_map = dict(keyword_scores) if keyword_scores else {}

        # 计算综合分数
        reranked = []
        for item in documents:
            if is_search_result:
                doc = item.document
                vector_score = item.score
            else:
                doc, vector_score = item

            # 关键词分数
            keyword_score = keyword_map.get(doc.id, 0.0)

            # 时间分数（基于创建时间）
            recency_score = self._calculate_recency_score(doc.metadata)

            # 综合分数
            combined_score = (
                self.weights["vector_score"] * vector_score
                + self.weights["keyword_score"] * keyword_score
                + self.weights["recency_score"] * recency_score
            )

            if is_search_result:
                # 保留原始的rerank_score，如果有的话
                reranked.append(
                    SearchResult(
                        document=doc, score=vector_score, rerank_score=combined_score
                    )
                )
            else:
                reranked.append((doc, combined_score))

        # 排序
        if is_search_result:
            reranked.sort(
                key=lambda x: x.rerank_score if x.rerank_score is not None else x.score,
                reverse=True,
            )
        else:
            reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """计算时间分数"""
        import time

        created_at = metadata.get("created_at", 0)
        if not created_at:
            return 0.5

        # 时间衰减：越新的文档分数越高
        days_old = (time.time() - created_at) / (24 * 3600)
        return max(0.1, 1.0 - (days_old / 365))  # 一年后衰减到0.1


# API重排序服务
class APIReranker:
    """API重排序器"""

    def __init__(self, config: Optional[APIRerankConfig] = None):
        self.api_service = APIRerankService(config)

    async def initialize(self):
        """初始化API重排序器"""
        await self.api_service.initialize()

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]] | List[SearchResult],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]] | List[SearchResult]:
        """使用API服务重排序"""
        # API服务可能只支持Tuple[Document, float]格式
        # 如果输入是SearchResult，需要转换
        if documents and isinstance(documents[0], SearchResult):
            converted_docs = [(item.document, item.score) for item in documents]
            result = await self.api_service.rerank(query, converted_docs, top_k)
            # 将结果转换回SearchResult
            return [
                SearchResult(document=doc, score=score, rerank_score=score)
                for doc, score in result
            ]
        else:
            return await self.api_service.rerank(query, documents, top_k)

    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return self.api_service.get_config()


# 增强的混合重排序器
class EnhancedHybridReranker:
    """增强的混合重排序器，支持API和简单重排序"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # API重排序配置
        self.api_reranker = APIReranker(APIRerankConfig(**config.get("api", {})))

        # 简单重排序
        self.simple_reranker = SimpleReranker()

        # 策略配置
        self.strategy = config.get("strategy", "auto")  # auto, api, simple

    async def initialize(self):
        """初始化所有重排序器"""
        await self.api_reranker.initialize()

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]] | List[SearchResult],
        keyword_scores: Optional[List[Tuple[str, float]]] = None,
        top_k: int = 5,
        strategy: str = None,
    ) -> List[Tuple[Document, float]] | List[SearchResult]:
        """增强的混合重排序"""
        if not documents:
            return []

        strategy = strategy or self.strategy

        if strategy == "api":
            # 优先使用API重排序
            return await self.api_reranker.rerank(query, documents, top_k)

        elif strategy == "simple":
            # 使用简单重排序
            return await self.simple_reranker.rerank(
                query, documents, keyword_scores, top_k
            )

        else:  # auto
            # 自动选择最佳策略
            api_config = self.api_reranker.api_service.config
            if api_config.validate():
                # API配置有效，优先使用API
                return await self.api_reranker.rerank(query, documents, top_k)
            else:
                # 降级到简单重排序
                return await self.simple_reranker.rerank(
                    query, documents, keyword_scores, top_k
                )

    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "strategy": self.strategy,
            "api_config": self.api_reranker.get_config(),
            "simple_reranker_weights": self.simple_reranker.weights,
        }


# 重排序策略工厂（更新版）
class RerankStrategyFactory:
    """重排序策略工厂（支持API重排序）"""

    _strategies = {
        "simple": SimpleReranker,
        "api": APIReranker,
        "enhanced_hybrid": EnhancedHybridReranker,
    }

    @classmethod
    def create(cls, strategy: str, config: Optional[Any] = None):
        """创建重排序策略"""
        if strategy not in cls._strategies:
            raise ValueError(f"不支持的重排序策略: {strategy}")

        return cls._strategies[strategy](config)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有支持的策略"""
        return list(cls._strategies.keys())


# 异步重排序包装器
class AsyncRerankWrapper:
    """异步重排序包装器"""

    def __init__(self, reranker):
        self.reranker = reranker

    async def rerank(self, *args, **kwargs):
        """异步重排序"""
        if hasattr(self.reranker, "initialize"):
            await self.reranker.initialize()

        return await self.reranker.rerank(*args, **kwargs)
