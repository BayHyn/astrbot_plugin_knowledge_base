# astrbot_plugin_knowledge_base/core/config_manager.py
from dataclasses import dataclass
from typing import Optional
from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent
from astrbot.core.config.default import VERSION


@dataclass
class KBConfig:
    """知识库配置统一模型"""

    # 数据库配置
    vector_db_type: str = "faiss"
    faiss_db_path: str = "faiss_data"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""

    # Embedding 配置
    embedding_api_url: str = ""
    embedding_api_key: str = ""
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dimension: int = 1024

    # RAG 配置
    search_top_k: int = 3
    min_similarity_score: float = 0.5
    insertion_method: str = "prepend_prompt"
    context_template: str = "这是相关的知识库信息，请参考这些信息来回答用户的问题：\n{retrieved_contexts}"
    max_insert_length: int = 200000

    # 文本处理
    chunk_size: int = 300
    chunk_overlap: int = 100

    # 默认知识库
    default_collection: Optional[str] = None

    # 其他配置
    auto_create_collection: bool = True
    llm_model: str = ""

    @classmethod
    def from_plugin_config(cls, config: AstrBotConfig) -> "KBConfig":
        """从插件配置加载"""
        # 获取 llm_rag_settings 配置块
        llm_rag_config = config.get("llm_rag_settings", {})

        return cls(
            # 数据库配置
            vector_db_type=config.get("vector_db_type", "faiss"),
            faiss_db_path=config.get("faiss_db_subpath", "faiss_data"),
            milvus_host=config.get("milvus_host", "localhost"),
            milvus_port=config.get("milvus_port", 19530),
            milvus_user=config.get("milvus_user", ""),
            milvus_password=config.get("milvus_password", ""),
            # Embedding 配置
            embedding_api_url=config.get("embedding_api_url", ""),
            embedding_api_key=config.get("embedding_api_key", ""),
            embedding_model_name=config.get("embedding_model_name", "text-embedding-3-small"),
            embedding_dimension=config.get("embedding_dimension", 1024),
            # RAG 配置
            search_top_k=config.get("search_top_k", 3),
            min_similarity_score=llm_rag_config.get("min_similarity_score", 0.5),
            insertion_method=llm_rag_config.get("insertion_method", "prepend_prompt"),
            context_template=llm_rag_config.get(
                "context_template",
                "这是相关的知识库信息，请参考这些信息来回答用户的问题：\n{retrieved_contexts}",
            ),
            max_insert_length=llm_rag_config.get("max_insert_length", 200000),
            # 文本处理
            chunk_size=config.get("text_chunk_size", 300),
            chunk_overlap=config.get("text_chunk_overlap", 100),
            # 默认知识库（从插件配置读取，仅作为 fallback）
            default_collection=config.get("default_collection_name", None),
            # 其他配置
            auto_create_collection=config.get("auto_create_collection", True),
            llm_model=config.get("LLM_model", ""),
        )

    def validate(self):
        """配置验证"""
        errors = []

        if self.search_top_k < 1:
            errors.append("search_top_k must be >= 1")

        if not 0 <= self.min_similarity_score <= 1:
            errors.append("min_similarity_score must be in [0, 1]")

        if self.chunk_size < 1:
            errors.append("chunk_size must be >= 1")

        if self.chunk_overlap < 0:
            errors.append("chunk_overlap must be >= 0")

        if self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be < chunk_size")

        if self.insertion_method not in ["prepend_prompt", "system_prompt"]:
            errors.append(
                f"insertion_method must be 'prepend_prompt' or 'system_prompt', got '{self.insertion_method}'"
            )

        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

        logger.debug("配置验证通过")


@dataclass
class UserKBConfig:
    """用户级别的知识库配置（合并全局配置和用户偏好）"""

    base_config: KBConfig
    default_collection: Optional[str]

    @property
    def search_top_k(self) -> int:
        return self.base_config.search_top_k

    @property
    def min_similarity_score(self) -> float:
        return self.base_config.min_similarity_score

    @property
    def insertion_method(self) -> str:
        return self.base_config.insertion_method

    @property
    def context_template(self) -> str:
        return self.base_config.context_template

    @property
    def max_insert_length(self) -> int:
        return self.base_config.max_insert_length


class ConfigManager:
    """统一配置管理器"""

    def __init__(self, plugin_config: AstrBotConfig, context: Context):
        self.plugin_config = plugin_config
        self.context = context
        self.kb_config = KBConfig.from_plugin_config(plugin_config)

        # 验证配置
        try:
            self.kb_config.validate()
            logger.info("知识库配置加载并验证成功")
        except ValueError as e:
            logger.error(f"配置验证失败: {e}")
            raise

    def get_user_kb_config(
        self, event: AstrMessageEvent, user_default_collection: Optional[str] = None
    ) -> UserKBConfig:
        """
        获取用户级别的配置（合并全局配置和用户偏好）

        Args:
            event: 消息事件
            user_default_collection: 用户的默认知识库（由 UserPrefsHandler 提供）

        Returns:
            UserKBConfig: 用户级别的配置对象
        """
        user_key = event.unified_msg_origin

        # 优先级: 会话偏好 > 用户配置 > 全局配置
        default_collection = user_default_collection

        # 如果没有会话偏好，尝试从 AstrBot 4.0+ 配置读取
        if default_collection is None and VERSION >= "4.0.0":
            user_cfg = self.context.get_config(umo=user_key)
            cfg_collection = user_cfg.get("default_kb_collection")
            # 将空字符串转换为 None
            if cfg_collection and cfg_collection != "":
                default_collection = cfg_collection

        # 如果还是 None，使用全局配置的默认值
        if default_collection is None:
            default_collection = self.kb_config.default_collection

        return UserKBConfig(
            base_config=self.kb_config,
            default_collection=default_collection,
        )

    def update_embedding_config(self, dimension: int, model_name: str):
        """更新 Embedding 配置（用于适配器插件）"""
        self.kb_config.embedding_dimension = dimension
        self.kb_config.embedding_model_name = model_name
        logger.info(
            f"Embedding 配置已更新: dimension={dimension}, model_name={model_name}"
        )
