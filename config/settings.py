from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class LLMSettings(BaseModel):
    """用于文件解析的LLM提供商设置"""

    enable_llm_parser: bool = Field(True, description="启用LLM解析复杂文件，如图片")
    provider: Optional[str] = Field(None, description="来自AstrBot的LLM提供商ID")


class RerankSettings(BaseModel):
    """重排序配置"""

    strategy: str = Field(
        "auto", description="重排序策略: auto, api, simple"
    )
    api_provider: str = Field("cohere", description="重排序 API 提供商")
    api_key: Optional[str] = Field(None, description="重排序 API 密钥")
    api_url: Optional[str] = Field(None, description="重排序 API URL")
    model_name: Optional[str] = Field(None, description="重排序模型名称")
    enable_cache: bool = Field(True, description="启用API重排序缓存")
    cache_ttl: int = Field(3600, description="缓存过期时间（秒）")
    timeout: int = Field(30, description="API超时时间（秒）")
    max_retries: int = Field(3, description="最大重试次数")


class PluginSettings(BaseModel):
    """主插件设置"""

    default_collection_name: str = Field("general", description="默认知识库集合名称")
    faiss_db_subpath: str = Field("faiss_data", description="Faiss索引存储路径")
    text_chunk_size: int = Field(300, description="文本分块大小")
    text_chunk_overlap: int = Field(100, description="文本分块重叠大小")
    search_top_k: int = Field(3, description="搜索返回的结果数量")
    auto_create_collection: bool = Field(True, description="自动创建知识库集合")
    enable_kb_llm_enhancement: bool = Field(
        True, description="使用知识库内容增强LLM请求"
    )
    kb_llm_search_top_k: int = Field(3, description="为LLM增强检索的文档数量")
    kb_llm_insertion_method: str = Field(
        "prepend_prompt",
        description="如何插入知识库内容: prepend_prompt 或 system_prompt",
    )
    kb_llm_min_similarity_score: float = Field(
        0.5, description="认为文档相关的最低相似度分数"
    )
    kb_llm_context_template: str = Field(
        "以下是可能相关的知识库内容：\n---\n{retrieved_contexts}\n---\n请根据以上信息回答我的问题。",
        description="用于包装知识库上下文的模板",
    )

    llm_parser: LLMSettings = Field(default_factory=LLMSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)

    @classmethod
    def from_astrbot_config(cls, config: Dict[str, Any]) -> "PluginSettings":
        """从AstrBot的配置字典创建设置"""
        def get_config_value(key: str, default_value=None):
            prefixed_key = f"astrbot_plugin_knowledge_base_{key}"
            if prefixed_key in config:
                return config[prefixed_key]
            return config.get(key, default_value)
        
        # 处理重排序配置
        rerank_config = get_config_value("rerank_config", {})
        
        return cls(
            default_collection_name=get_config_value("default_collection_name", "general"),
            faiss_db_subpath=get_config_value("faiss_db_subpath", "faiss_data"),
            text_chunk_size=get_config_value("text_chunk_size", 300),
            text_chunk_overlap=get_config_value("text_chunk_overlap", 100),
            search_top_k=get_config_value("search_top_k", 3),
            auto_create_collection=get_config_value("auto_create_collection", True),
            enable_kb_llm_enhancement=get_config_value("enable_kb_llm_enhancement", True),
            kb_llm_search_top_k=get_config_value("kb_llm_search_top_k", 3),
            kb_llm_insertion_method=get_config_value("kb_llm_insertion_method", "prepend_prompt"),
            kb_llm_min_similarity_score=get_config_value("kb_llm_min_similarity_score", 0.5),
            kb_llm_context_template=get_config_value(
                "kb_llm_context_template",
                "以下是可能相关的知识库内容：\n---\n{retrieved_contexts}\n---\n请根据以上信息回答我的问题。",
            ),
            llm_parser=LLMSettings(
                enable_llm_parser=get_config_value("enable_llm_parser", True),
                provider=get_config_value("llm_parser_provider"),
            ),
            rerank=RerankSettings(
                strategy=rerank_config.get("strategy", "auto"),
                api_provider=rerank_config.get("api_provider", "cohere"),
                api_key=rerank_config.get("api_key"),
                api_url=rerank_config.get("api_url"),
                model_name=rerank_config.get("model_name"),
                enable_cache=rerank_config.get("enable_cache", True),
                cache_ttl=rerank_config.get("cache_ttl", 3600),
                timeout=rerank_config.get("timeout", 30),
                max_retries=rerank_config.get("max_retries", 3),
            ),
        )
