from typing import List, Optional, Tuple
from astrbot.api import logger
from astrbot.api.star import Context


class EmbeddingUtil:
    def __init__(self, provider_context: Context = None):
        """
        初始化嵌入工具，使用AstrBot内置的提供商系统
        
        Args:
            provider_context: AstrBot的上下文，用于获取提供商
        """
        self.context = provider_context
        self._default_provider = None
        self._embedding_dimension = None

    def _get_provider(self):
        """获取默认的embedding提供商"""
        if not self.context:
            logger.error("EmbeddingUtil: 上下文未设置，无法获取提供商")
            return None
            
        if not self._default_provider:
            # 尝试获取默认的提供商
            providers = self.context.get_all_providers()
            if providers:
                # 选择第一个可用的提供商
                self._default_provider = providers[0]
                logger.info(f"EmbeddingUtil: 使用提供商 {self._default_provider.provider_name}")
            else:
                logger.error("EmbeddingUtil: 没有可用的提供商")
                return None
                
        return self._default_provider

    async def get_embedding_async(self, text: str) -> Optional[List[float]]:
        """获取单个文本的 embedding"""
        provider = self._get_provider()
        if not provider:
            logger.error("嵌入提供商未初始化")
            return None
            
        if not text or not text.strip():
            logger.warning("输入文本为空或仅包含空白")
            return None
            
        try:
            # 使用AstrBot提供商的embedding接口
            # 注意：这里需要根据实际的AstrBot提供商接口进行调整
            if hasattr(provider, 'get_embedding'):
                embedding = await provider.get_embedding(text)
                if embedding:
                    return embedding
            else:
                logger.warning(f"提供商 {provider.provider_name} 不支持embedding功能")
                return None
                
        except Exception as e:
            logger.error(f"获取嵌入时发生错误: {e}", exc_info=True)
            return None

    async def get_embeddings_async(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """获取多个文本的 embedding"""
        if not texts:
            return []
            
        provider = self._get_provider()
        if not provider:
            logger.error("嵌入提供商未初始化")
            return [None] * len(texts)

        valid_texts_with_indices = [
            (i, text) for i, text in enumerate(texts) if text and text.strip()
        ]
        if not valid_texts_with_indices:
            return [None] * len(texts)

        final_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        # 批量处理
        try:
            if hasattr(provider, 'get_embeddings'):
                # 如果提供商支持批量embedding
                batch_texts = [text for _, text in valid_texts_with_indices]
                embeddings = await provider.get_embeddings(batch_texts)
                
                if embeddings and len(embeddings) == len(batch_texts):
                    for idx, (original_idx, _) in enumerate(valid_texts_with_indices):
                        final_embeddings[original_idx] = embeddings[idx]
            else:
                # 逐个处理
                for original_idx, text in valid_texts_with_indices:
                    embedding = await self.get_embedding_async(text)
                    final_embeddings[original_idx] = embedding
                    
        except Exception as e:
            logger.error(f"批量获取嵌入失败: {e}", exc_info=True)

        return final_embeddings

    async def close(self):
        """关闭资源"""
        # AstrBot提供商由框架管理，这里不需要特别处理
        pass

    def get_dimensions(self) -> Optional[int]:
        """获取 embedding 维度"""
        if self._embedding_dimension:
            return self._embedding_dimension
            
        provider = self._get_provider()
        if provider and hasattr(provider, 'get_embedding_dimension'):
            self._embedding_dimension = provider.get_embedding_dimension()
            return self._embedding_dimension
            
        # 默认维度（常见的embedding模型维度）
        logger.warning("无法获取embedding维度，使用默认值1536")
        return 1536
