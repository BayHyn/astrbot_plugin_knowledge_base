# astrbot_plugin_knowledge_base/core/domain.py
"""领域模型层 - 定义核心数据结构和接口"""
from dataclasses import dataclass
from typing import Protocol, Optional, Dict
from abc import ABC, abstractmethod


@dataclass
class CollectionMetadata:
    """知识库集合元数据"""

    collection_name: str
    version: int = 1
    emoji: str = "🙂"
    description: str = ""
    created_at: int = 0
    file_id: str = ""
    origin: str = "unknown"
    embedding_provider_id: Optional[str] = None
    rerank_provider_id: Optional[str] = None

    @classmethod
    def from_dict(cls, collection_name: str, data: Dict) -> "CollectionMetadata":
        """从字典创建元数据对象"""
        return cls(
            collection_name=collection_name,
            version=data.get("version", 1),
            emoji=data.get("emoji", "🙂"),
            description=data.get("description", ""),
            created_at=data.get("created_at", 0),
            file_id=data.get("file_id", ""),
            origin=data.get("origin", "unknown"),
            embedding_provider_id=data.get("embedding_provider_id"),
            rerank_provider_id=data.get("rerank_provider_id"),
        )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "version": self.version,
            "emoji": self.emoji,
            "description": self.description,
            "created_at": self.created_at,
            "file_id": self.file_id,
            "origin": self.origin,
            "embedding_provider_id": self.embedding_provider_id,
            "rerank_provider_id": self.rerank_provider_id,
        }


class CollectionMetadataRepository(Protocol):
    """集合元数据仓库接口 - 使用 Protocol 实现依赖倒置"""

    def get_metadata(self, collection_name: str) -> Optional[CollectionMetadata]:
        """获取指定集合的元数据"""
        ...

    def get_all_metadata(self) -> Dict[str, CollectionMetadata]:
        """获取所有集合的元数据"""
        ...

    def set_metadata(self, metadata: CollectionMetadata) -> None:
        """设置集合元数据"""
        ...

    def delete_metadata(self, collection_name: str) -> None:
        """删除集合元数据"""
        ...


class ProviderAccessor(Protocol):
    """提供商访问器接口 - 用于获取 AstrBot 提供商"""

    def get_provider_by_id(self, provider_id: str):
        """根据 ID 获取提供商实例"""
        ...
