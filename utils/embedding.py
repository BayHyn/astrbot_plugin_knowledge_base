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
        logger.debug(f"开始获取集合 '{collection_name}' 的embedding提供商")

        if not user_prefs_handler:
            logger.warning("user_prefs_handler为空，使用默认提供商")
            return self._get_default_embedding_provider()

        # 从集合元数据中获取提供商ID
        try:
            collection_metadata = user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            logger.debug(f"集合元数据: {collection_metadata}")

            collection_config = collection_metadata.get(collection_name, {})
            logger.debug(f"集合 '{collection_name}' 配置: {collection_config}")

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
            logger.info(f"尝试获取提供商: {astrbot_embedding_provider_id}")

            if not self.context:
                logger.error("self.context为空，无法获取提供商")
                return self._get_default_embedding_provider()

            try:
                provider = self.context.get_provider_by_id(
                    astrbot_embedding_provider_id
                )
                logger.debug(f"get_provider_by_id返回: {provider}")
                logger.debug(f"provider类型: {type(provider)}")

                if provider:
                    from astrbot.core.provider.provider import EmbeddingProvider

                    logger.debug(f"EmbeddingProvider类: {EmbeddingProvider}")
                    logger.debug(
                        f"isinstance检查: {isinstance(provider, EmbeddingProvider)}"
                    )

                    if isinstance(provider, EmbeddingProvider):
                        # 详细检查provider的配置
                        try:
                            provider_config = getattr(provider, "provider_config", {})
                            logger.info(
                                f"✅ 找到有效的embedding提供商: {astrbot_embedding_provider_id}"
                            )
                            logger.info(
                                f"   - 类型: {provider_config.get('type', 'unknown')}"
                            )
                            logger.info(
                                f"   - API Base: {provider_config.get('embedding_api_base', 'unknown')}"
                            )
                            logger.info(
                                f"   - 模型: {provider_config.get('embedding_model', 'unknown')}"
                            )

                            # TODO: 验证provider的关键方法是否存在
                            if hasattr(provider, "get_embedding"):
                                logger.debug("✅ provider有get_embedding方法")
                            else:
                                logger.error("❌ provider缺少get_embedding方法")

                            if hasattr(provider, "get_embeddings"):
                                logger.debug("✅ provider有get_embeddings方法")
                            else:
                                logger.warning("⚠️  provider缺少get_embeddings方法")

                        except Exception as e:
                            logger.error(f"检查provider配置时出错: {e}")

                        return provider
                    else:
                        logger.error(
                            f"❌ 提供商 {astrbot_embedding_provider_id} 不是 EmbeddingProvider 类型"
                        )
                        logger.error(f"   实际类型: {type(provider)}")
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
        logger.debug("开始获取默认embedding提供商")

        if not self.context:
            logger.error("EmbeddingUtil: 上下文未设置，无法获取提供商")
            return None

        if not self._default_provider:
            logger.debug("缓存中没有默认提供商，开始查找...")

            # 尝试获取所有embedding提供商
            try:
                providers = self.context.get_all_providers()
                logger.info(
                    f"从上下文获取到 {len(providers) if providers else 0} 个提供商"
                )

                if providers:
                    from astrbot.core.provider.provider import EmbeddingProvider

                    # 详细检查每个提供商
                    for i, provider in enumerate(providers):
                        logger.debug(f"检查提供商 {i}: {type(provider)}")

                        try:
                            provider_config = getattr(provider, "provider_config", {})
                            provider_id = provider_config.get("id", f"unknown_{i}")
                            provider_type = provider_config.get("type", "unknown")
                            logger.debug(
                                f"  - ID: {provider_id}, 类型: {provider_type}"
                            )

                            # 检查是否是EmbeddingProvider
                            if isinstance(provider, EmbeddingProvider):
                                logger.info(f"✅ 找到embedding提供商: {provider_id}")
                                self._default_provider = provider

                                # 详细记录选中的提供商信息
                                logger.info(f"📋 默认embedding提供商详情:")
                                logger.info(f"   - ID: {provider_id}")
                                logger.info(f"   - 类型: {provider_type}")
                                logger.info(
                                    f"   - API Base: {provider_config.get('embedding_api_base', 'N/A')}"
                                )
                                logger.info(
                                    f"   - 模型: {provider_config.get('embedding_model', 'N/A')}"
                                )
                                logger.info(
                                    f"   - 维度: {provider_config.get('embedding_dimensions', 'N/A')}"
                                )

                                # TODO: 测试provider是否可用
                                try:
                                    if hasattr(provider, "get_embedding"):
                                        logger.debug("✅ 提供商支持get_embedding方法")
                                    else:
                                        logger.error("❌ 提供商不支持get_embedding方法")

                                    if hasattr(provider, "get_embeddings"):
                                        logger.debug("✅ 提供商支持get_embeddings方法")
                                    else:
                                        logger.warning(
                                            "⚠️  提供商不支持get_embeddings方法"
                                        )

                                except Exception as e:
                                    logger.error(f"检查提供商方法时出错: {e}")

                                break
                            else:
                                logger.debug(f"  - 不是EmbeddingProvider类型，跳过")

                        except Exception as e:
                            logger.error(f"检查提供商 {i} 时出错: {e}")

                    if not self._default_provider:
                        logger.error("❌ 没有找到可用的embedding提供商")
                        logger.error("可用的提供商类型:")
                        for i, provider in enumerate(providers):
                            logger.error(f"  {i}: {type(provider)}")
                        return None
                else:
                    logger.error("❌ 上下文中没有任何提供商")
                    return None

            except Exception as e:
                logger.error(f"获取提供商列表时出错: {e}", exc_info=True)
                return None
        else:
            logger.debug("使用缓存的默认提供商")

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
        logger.debug(
            f"文本样本: {[text[:50] + '...' if len(text) > 50 else text for text in texts[:3]]}"
        )

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

        # TODO: 添加提供商信息日志用于调试
        try:
            provider_name = (
                provider.meta().id if hasattr(provider, "meta") else "unknown"
            )
            provider_type = (
                provider.provider_config.get("type", "unknown")
                if hasattr(provider, "provider_config")
                else "unknown"
            )
            logger.info(f"使用embedding提供商: {provider_name} (类型: {provider_type})")
        except Exception as e:
            logger.warning(f"无法获取提供商信息: {e}")

        # 过滤有效文本
        valid_texts_with_indices = [
            (i, text) for i, text in enumerate(texts) if text and text.strip()
        ]
        if not valid_texts_with_indices:
            return [None] * len(texts)

        logger.info(f"有效文本数量: {len(valid_texts_with_indices)}/{len(texts)}")
        final_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        try:
            # 检查提供商是否支持批量embedding
            if hasattr(provider, "get_embeddings"):
                # 使用批量接口，但需要处理批次大小限制
                batch_texts = [text for _, text in valid_texts_with_indices]

                # 设置批次大小限制（动态获取）
                max_batch_size = self._get_max_batch_size(provider)
                logger.debug(f"使用批次大小限制: {max_batch_size}")

                if len(batch_texts) <= max_batch_size:
                    # 如果文本数量不超过限制，直接处理
                    logger.info(f"单批处理 {len(batch_texts)} 个文本")

                    # TODO: 详细记录API调用前的状态
                    logger.debug(f"准备调用 provider.get_embeddings()...")
                    logger.debug(f"Provider类型: {type(provider)}")
                    logger.debug(f"调用方法: get_embeddings")
                    logger.debug(
                        f"文本样本长度: {[len(text) for text in batch_texts[:3]]}"
                    )

                    batch_start = time.time()
                    try:
                        logger.info(
                            f"🌐 开始API调用: get_embeddings({len(batch_texts)}个文本)"
                        )

                        # TODO: 检查是否是真实的网络调用
                        # 记录调用前的详细信息
                        logger.debug(f"调用详情:")
                        logger.debug(f"  - Provider: {type(provider).__name__}")
                        logger.debug(
                            f"  - API Base: {getattr(provider, 'provider_config', {}).get('embedding_api_base', 'N/A')}"
                        )
                        logger.debug(
                            f"  - 模型: {getattr(provider, 'provider_config', {}).get('embedding_model', 'N/A')}"
                        )

                        embeddings = await provider.get_embeddings(batch_texts)
                        logger.info(f"🌐 API调用完成")
                    except Exception as api_e:
                        logger.error(f"❌ API调用失败: {api_e}", exc_info=True)
                        embeddings = None

                    batch_end = time.time()
                    api_time = batch_end - batch_start
                    logger.info(f"单批API调用耗时: {api_time:.2f}秒")

                    # 验证返回结果
                    if embeddings:
                        logger.info(f"API返回: {len(embeddings)} 个embedding")
                        if len(embeddings) == len(batch_texts):
                            # TODO: 验证embedding质量和维度
                            sample_embedding = embeddings[0] if embeddings else None
                            if sample_embedding:
                                logger.info(
                                    f"返回embedding样本维度: {len(sample_embedding)}"
                                )
                                logger.debug(
                                    f"embedding样本前5个值: {sample_embedding[:5]}"
                                )

                                # 检查embedding是否看起来是真实的（不是全零或全一）
                                import numpy as np

                                embedding_array = np.array(sample_embedding)
                                is_zeros = np.allclose(embedding_array, 0.0)
                                is_ones = np.allclose(embedding_array, 1.0)
                                std_dev = np.std(embedding_array)

                                if is_zeros:
                                    logger.error("⚠️  embedding全为0，可能是mock数据！")
                                elif is_ones:
                                    logger.error("⚠️  embedding全为1，可能是mock数据！")
                                elif std_dev < 0.01:
                                    logger.warning(
                                        f"⚠️  embedding方差很小({std_dev:.6f})，可能是假数据"
                                    )
                                else:
                                    logger.info(
                                        f"✅ embedding看起来正常，标准差: {std_dev:.4f}"
                                    )

                            for idx, (original_idx, _) in enumerate(
                                valid_texts_with_indices
                            ):
                                final_embeddings[original_idx] = embeddings[idx]
                        else:
                            logger.error(
                                f"❌ 批量API返回异常: 期望{len(batch_texts)}个embedding, 实际{len(embeddings)}个"
                            )
                    else:
                        logger.error(f"❌ API调用返回空结果")

                    # API耗时异常检测
                    if api_time < 0.1 and len(batch_texts) > 5:
                        logger.error(
                            f"🚨 API调用时间异常短: {api_time:.3f}秒处理{len(batch_texts)}个文本，这不正常！"
                        )
                        logger.error("   可能的原因:")
                        logger.error("   1. Provider返回了缓存数据而不是真实API调用")
                        logger.error("   2. Provider使用了本地模型而不是远程API")
                        logger.error("   3. Provider的实现有问题")
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
                            chunk_start = time.time()
                            chunk_embeddings = await provider.get_embeddings(
                                batch_chunk
                            )
                            chunk_end = time.time()
                            logger.info(
                                f"批次 {batch_num} API调用耗时: {chunk_end - chunk_start:.2f}秒"
                            )

                            if chunk_embeddings:
                                all_embeddings.extend(chunk_embeddings)
                                logger.debug(
                                    f"批次 {batch_num} 成功返回 {len(chunk_embeddings)} 个embedding"
                                )
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
                    individual_start = time.time()
                    embedding = await provider.get_embedding(text)
                    individual_end = time.time()
                    logger.debug(
                        f"单个embedding调用耗时: {individual_end - individual_start:.2f}秒"
                    )
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

        if total_time < 1.0 and len(texts) > 10:
            logger.warning(
                f"⚠️  处理 {len(texts)} 个文本仅耗时 {total_time:.2f}秒，这可能表明embedding没有真正调用API服务！"
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
