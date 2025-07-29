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
            logger.warning("EmbeddingUtil: API key or URL is not configured. Client not created.")
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
            logger.error("Embedding client is not initialized.")
            return None
        if not text or not text.strip():
            logger.warning("Input text is empty or contains only whitespace.")
            return None
        try:
            response = await self.client.embeddings.create(
                input=text, model=self.settings.model_name
            )
            if response.data and hasattr(response.data[0], "embedding"):
                return response.data[0].embedding
            else:
                logger.error(f"Failed to get embedding, invalid API response: {response}")
                return None
        except APIStatusError as e:
            logger.error(f"API request failed (Status Error): {e.status_code}, {e.message}")
            return None
        except APIConnectionError as e:
            logger.error(f"API connection failed: {e}")
            return None
        except RateLimitError as e:
            logger.error(f"API rate limit exceeded: {e}")
            return None
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            logger.error(f"An unknown error occurred while getting embedding: {e}", exc_info=True)
            return None

    async def get_embeddings_async(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """获取多个文本的 embedding (使用 OpenAI 批量接口)，每 10 条文本分批请求"""
        if not self.client:
            logger.error("Embedding client is not initialized.")
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
                    logger.error(f"Batch {i // batch_size + 1}: Mismatch in response count.")
            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1}: Failed to get embeddings: {e}")

        return final_embeddings

    async def close(self):
        """关闭 OpenAI 客户端"""
        if self.client:
            await self.client.close()

    def get_dimensions(self) -> Optional[int]:
        """获取 embedding 维度"""
        return self.settings.dimensions
