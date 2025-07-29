from typing import List, Optional, Tuple
from astrbot.api import logger
from openai import (
    AsyncOpenAI,
    APIError,
    APIStatusError,
    APIConnectionError,
    RateLimitError,
)

from ..config.settings import EmbeddingSettings


class EmbeddingUtil:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self.client = self._create_client()

    def _create_client(self) -> Optional[AsyncOpenAI]:
        if not self.settings.api_key or not self.settings.api_url:
            logger.warning("EmbeddingUtil: API密钥或URL未配置。客户端未创建。")
            return None
        
        base_url = self.settings.api_url
        if base_url.endswith("/embeddings"):
            base_url = base_url[:-len("/embeddings")]

        return AsyncOpenAI(
            api_key=self.settings.api_key,
            base_url=base_url,
            timeout=30.0,
        )

    async def get_embedding_async(self, text: str) -> Optional[List[float]]:
        """获取单个文本的 embedding"""
        if not self.client:
            logger.error("嵌入客户端未初始化。")
            return None
        if not text or not text.strip():
            logger.warning("输入文本为空或仅包含空白。")
            return None
        try:
            response = await self.client.embeddings.create(
                input=text, model=self.settings.model_name
            )
            if response.data and hasattr(response.data[0], "embedding"):
                return response.data[0].embedding
            else:
                logger.error(f"获取嵌入失败，无效的API响应: {response}")
                return None
        except APIStatusError as e:
            logger.error(f"API请求失败 (状态错误): {e.status_code}, {e.message}")
            return None
        except APIConnectionError as e:
            logger.error(f"API连接失败: {e}")
            return None
        except RateLimitError as e:
            logger.error(f"API速率限制已超出: {e}")
            return None
        except APIError as e:
            logger.error(f"OpenAI API错误: {e}")
            return None
        except Exception as e:
            logger.error(f"获取嵌入时发生未知错误: {e}", exc_info=True)
            return None

    async def get_embeddings_async(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """获取多个文本的 embedding (使用 OpenAI 批量接口)，每 10 条文本分批请求"""
        if not self.client:
            logger.error("嵌入客户端未初始化。")
            return [None] * len(texts)
        if not texts:
            return []

        valid_texts_with_indices = [
            (i, text) for i, text in enumerate(texts) if text and text.strip()
        ]
        if not valid_texts_with_indices:
            return [None] * len(texts)

        final_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        batch_size = 10

        for i in range(0, len(valid_texts_with_indices), batch_size):
            batch_indices_texts = valid_texts_with_indices[i:i + batch_size]
            batch_texts = [text for _, text in batch_indices_texts]

            try:
                response = await self.client.embeddings.create(
                    input=batch_texts, model=self.settings.model_name
                )
                if response.data and len(response.data) == len(batch_texts):
                    for idx, (original_idx, _) in enumerate(batch_indices_texts):
                        final_embeddings[original_idx] = response.data[idx].embedding
                else:
                    logger.error(f"批处理 {i // batch_size + 1}: 响应计数不匹配。")
            except Exception as e:
                logger.error(f"批处理 {i // batch_size + 1}: 获取嵌入失败: {e}")

        return final_embeddings

    async def close(self):
        """关闭 OpenAI 客户端"""
        if self.client:
            await self.client.close()

    def get_dimensions(self) -> Optional[int]:
        """获取 embedding 维度"""
        return self.settings.dimensions
