from typing import List, Optional
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

    def _get_embedding_provider_for_collection(
        self, user_prefs_handler, collection_name: str
    ):
        """根据集合配置获取对应的embedding提供商"""
        if not user_prefs_handler:
            logger.warning("user_prefs_handler为空，使用默认提供商")
            return self._get_default_embedding_provider()

        # 从集合元数据中获取提供商ID
        try:
            collection_metadata = user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            collection_config = collection_metadata.get(collection_name, {})
            astrbot_embedding_provider_id = collection_config.get(
                "embedding_provider_id", None
            )
            logger.info(
                f"集合 '{collection_name}' 指定的embedding_provider_id: '{astrbot_embedding_provider_id}'"
            )
        except Exception as e:
            logger.error(f"获取集合元数据时出错: {e}")
            return self._get_default_embedding_provider()

        if astrbot_embedding_provider_id:
            if not self.context:
                logger.error("self.context为空，无法获取提供商")
                return self._get_default_embedding_provider()

            try:
                provider = self.context.get_provider_by_id(
                    astrbot_embedding_provider_id
                )
                if provider:
                    from astrbot.core.provider.provider import EmbeddingProvider

                    if isinstance(provider, EmbeddingProvider):
                        logger.info(
                            f"✅ 找到有效的embedding提供商: {astrbot_embedding_provider_id}"
                        )
                        return provider
                    else:
                        logger.error(
                            f"❌ 提供商 {astrbot_embedding_provider_id} 不是 EmbeddingProvider 类型, 实际类型: {type(provider)}"
                        )
                else:
                    logger.error(
                        f"❌ 未找到ID为 '{astrbot_embedding_provider_id}' 的提供商"
                    )

            except Exception as e:
                logger.error(f"获取提供商时发生异常: {e}", exc_info=True)
        else:
            logger.warning("集合配置中没有指定embedding_provider_id")

        # 兜底：使用默认的embedding提供商
        logger.info("回退到默认embedding提供商")
        return self._get_default_embedding_provider()

    def _get_default_embedding_provider(self):
        """获取默认的embedding提供商（第一个可用的）"""
        if not self.context:
            logger.error("EmbeddingUtil: 上下文未设置，无法获取提供商")
            return None

        if not self._default_provider:
            try:
                providers = self.context.get_all_providers()
                if providers:
                    from astrbot.core.provider.provider import EmbeddingProvider

                    for provider in providers:
                        if isinstance(provider, EmbeddingProvider):
                            provider_id = getattr(provider, "provider_config", {}).get(
                                "id", "unknown"
                            )
                            logger.info(f"✅ 找到默认embedding提供商: {provider_id}")
                            self._default_provider = provider
                            break

                    if not self._default_provider:
                        logger.error("❌ 没有找到可用的embedding提供商")
                else:
                    logger.error("❌ 上下文中没有任何提供商")
            except Exception as e:
                logger.error(f"获取提供商列表时出错: {e}", exc_info=True)
                return None

        return self._default_provider

    async def get_embedding_async(
        self, text: str, collection_name: str = None, user_prefs_handler=None
    ) -> Optional[List[float]]:
        """获取单个文本的 embedding"""
        if not text or not text.strip():
            logger.warning("输入文本为空或仅包含空白")
            return None

        # 获取对应的提供商
        if collection_name and user_prefs_handler:
            provider = self._get_embedding_provider_for_collection(
                user_prefs_handler, collection_name
            )
        else:
            provider = self._get_default_embedding_provider()

        if not provider:
            logger.error("无法获取embedding提供商")
            return None

        try:
            # 使用AstrBot提供商的embedding接口
            embedding = await provider.get_embedding(text)
            if embedding:
                return embedding
            else:
                logger.error("提供商返回空的embedding")
                return None

        except Exception as e:
            logger.error(f"获取嵌入时发生错误: {e}", exc_info=True)
            return None

    def _get_max_batch_size(self, provider) -> int:
        """获取提供商的最大批次大小"""
        # TODO: 未来需要实现更智能的批次大小检测
        # - 从提供商配置或API文档中获取准确的批次限制
        # - 支持动态调整批次大小以优化性能
        # - 考虑不同模型的具体限制差异

        # 尝试从提供商获取批次大小限制
        if hasattr(provider, "max_batch_size"):
            # TODO: 验证提供商返回的批次大小是否可靠
            return min(provider.max_batch_size, 32)  # 强制不超过32

        # 根据提供商类型设置默认批次大小
        try:
            provider_type = provider.provider_config.get("type", "").lower()
            # TODO: 根据实际测试结果调整各提供商的最优批次大小
            if "openai" in provider_type:
                return 32  # 保守设置，避免413错误
            elif "claude" in provider_type:
                return 24  # Claude 的保守批次大小
            elif "huggingface" in provider_type:
                return 16  # HuggingFace 的保守批次大小
            else:
                return 16  # 默认最保守批次大小
        except:
            return 16  # 如果无法确定类型，使用最保守的默认值

    async def get_embeddings_async(
        self, texts: List[str], collection_name: str = None, user_prefs_handler=None
    ) -> List[Optional[List[float]]]:
        """获取多个文本的 embedding

        TODO: 未来需要优化的功能点
        - 实现缓存机制，避免重复计算相同文本的embedding
        - 添加文本预处理和长度限制检查
        - 支持流式处理大批量文本
        - 实现更智能的错误恢复和重试策略
        - 添加性能监控和统计功能
        """
        import time

        start_time = time.time()

        if not texts:
            return []

        logger.info(f"开始获取 {len(texts)} 个文本的embedding")

        # 获取对应的提供商
        if collection_name and user_prefs_handler:
            provider = self._get_embedding_provider_for_collection(
                user_prefs_handler, collection_name
            )
        else:
            provider = self._get_default_embedding_provider()

        if not provider:
            logger.error("无法获取embedding提供商")
            return [None] * len(texts)

        # 过滤有效文本
        valid_texts_with_indices = [
            (i, text) for i, text in enumerate(texts) if text and text.strip()
        ]
        if not valid_texts_with_indices:
            return [None] * len(texts)

        final_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        try:
            # 检查提供商是否支持批量embedding
            if hasattr(provider, "get_embeddings"):
                # 使用批量接口，但需要处理批次大小限制
                batch_texts = [text for _, text in valid_texts_with_indices]

                # 设置批次大小限制（动态获取）
                max_batch_size = self._get_max_batch_size(provider)
                if len(batch_texts) <= max_batch_size:
                    # 如果文本数量不超过限制，直接处理
                    logger.info(f"单批处理 {len(batch_texts)} 个文本")
                    batch_start = time.time()
                    try:
                        embeddings = await provider.get_embeddings(batch_texts)
                    except Exception as api_e:
                        logger.error(f"API调用失败: {api_e}", exc_info=True)
                        embeddings = None
                    api_time = time.time() - batch_start
                    logger.info(f"单批API调用耗时: {api_time:.2f}秒")

                    # 验证返回结果
                    if embeddings:
                        if len(embeddings) == len(batch_texts):
                            for idx, (original_idx, _) in enumerate(
                                valid_texts_with_indices
                            ):
                                final_embeddings[original_idx] = embeddings[idx]
                        else:
                            logger.error(
                                f"批量API返回异常: 期望{len(batch_texts)}个embedding, 实际{len(embeddings)}个"
                            )
                    else:
                        logger.error(f"API调用返回空结果")
                else:
                    # 分批处理
                    # TODO: 未来优化分批处理逻辑
                    # - 实现并行批处理以提高性能
                    # - 添加重试机制处理临时网络错误
                    # - 实现智能批次大小调整（根据成功率动态调整）
                    # - 添加进度回调支持
                    logger.info(
                        f"文本数量 {len(batch_texts)} 超过批次限制 {max_batch_size}，将分批处理"
                    )

                    all_embeddings = []
                    total_batches = (
                        len(batch_texts) + max_batch_size - 1
                    ) // max_batch_size

                    for i in range(0, len(batch_texts), max_batch_size):
                        batch_chunk = batch_texts[i : i + max_batch_size]
                        batch_num = i // max_batch_size + 1
                        logger.info(
                            f"处理批次 {batch_num}/{total_batches}: {len(batch_chunk)} 个文本"
                        )

                        try:
                            chunk_embeddings = await provider.get_embeddings(
                                batch_chunk
                            )
                            if chunk_embeddings:
                                all_embeddings.extend(chunk_embeddings)
                            else:
                                # 如果批次失败，用None填充
                                all_embeddings.extend([None] * len(batch_chunk))
                                logger.warning(
                                    f"批次 {batch_num}/{total_batches} 处理失败：返回空结果"
                                )
                        except Exception as e:
                            # TODO: 添加更细粒度的错误处理
                            # - 区分不同类型的API错误（限流、配额、网络等）
                            # - 实现指数退避重试策略
                            # - 支持部分批次失败时的恢复机制
                            logger.error(
                                f"批次 {batch_num}/{total_batches} 处理失败: {e}"
                            )
                            all_embeddings.extend([None] * len(batch_chunk))

                    # 将结果分配到最终数组
                    if len(all_embeddings) == len(batch_texts):
                        for idx, (original_idx, _) in enumerate(
                            valid_texts_with_indices
                        ):
                            final_embeddings[original_idx] = all_embeddings[idx]
                    else:
                        logger.error(
                            f"嵌入结果数量不匹配: 期望 {len(batch_texts)}, 实际 {len(all_embeddings)}"
                        )
            else:
                # 逐个处理
                logger.info("提供商不支持批量处理，将逐个获取嵌入")
                for original_idx, text in valid_texts_with_indices:
                    try:
                        embedding = await provider.get_embedding(text)
                        final_embeddings[original_idx] = embedding
                    except Exception as e:
                        logger.error(f"获取单个文本嵌入失败: {e}")
                        final_embeddings[original_idx] = None

        except Exception as e:
            logger.error(f"批量获取嵌入失败: {e}", exc_info=True)
            # 如果批量处理完全失败，尝试逐个处理
            logger.info("批量处理失败，尝试逐个获取嵌入")
            for original_idx, text in valid_texts_with_indices:
                try:
                    embedding = await provider.get_embedding(text)
                    final_embeddings[original_idx] = embedding
                except Exception as individual_e:
                    logger.error(f"获取单个文本嵌入失败: {individual_e}")
                    final_embeddings[original_idx] = None

        # 最终统计
        total_time = time.time() - start_time
        successful_embeddings = len([e for e in final_embeddings if e is not None])
        logger.info(
            f"Embedding生成完成: {successful_embeddings}/{len(texts)} 个成功, 总耗时: {total_time:.2f}秒"
        )

        return final_embeddings

    def get_dimensions(
        self, collection_name: str = None, user_prefs_handler=None
    ) -> Optional[int]:
        """获取 embedding 维度"""
        # 获取对应的提供商
        if collection_name and user_prefs_handler:
            provider = self._get_embedding_provider_for_collection(
                user_prefs_handler, collection_name
            )
        else:
            provider = self._get_default_embedding_provider()

        if provider and hasattr(provider, "get_dim"):
            try:
                return provider.get_dim()
            except Exception as e:
                logger.error(f"获取embedding维度失败: {e}")

        # 默认维度（常见的embedding模型维度）
        logger.warning("无法获取embedding维度，使用默认值1536")
        return 1536

    async def close(self):
        """关闭资源"""
        # AstrBot提供商由框架管理，这里不需要特别处理
        pass
