from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class EmbeddingSettings(BaseModel):
    """嵌入模型设置"""

    provider: str = Field("openai", description="嵌入服务的提供商, 例如 'openai'")
    api_url: Optional[str] = Field(None, description="嵌入服务的 API URL")
    api_key: Optional[str] = Field(None, description="嵌入服务的 API 密钥")
    model_name: str = Field("text-embedding-ada-002", description="嵌入模型的名称")
    dimensions: Optional[int] = Field(
        None, description="嵌入维度, 如果未设置则自动检测"
    )


class LLMSettings(BaseModel):
    """用于文件解析的LLM提供商设置"""

    enable_llm_parser: bool = Field(True, description="启用LLM解析复杂文件，如图片")
    provider: Optional[str] = Field(None, description="来自AstrBot的LLM提供商ID")


class RerankSettings(BaseModel):
    """重排序配置"""

    strategy: str = Field(
        "auto", description="重排序策略: auto, api, cross_encoder, simple"
    )
    api_provider: str = Field("cohere", description="重排序 API 提供商")
    api_key: Optional[str] = Field(None, description="重排序 API 密钥")
    api_url: Optional[str] = Field(None, description="重排序 API URL")


class PluginSettings(BaseModel):
    """主插件设置"""

    default_collection_name: str = Field("general", description="新用户的默认集合名称")
    auto_create_collection: bool = Field(True, description="如果集合不存在，则自动创建")
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

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm_parser: LLMSettings = Field(default_factory=LLMSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)

    @classmethod
    def from_astrbot_config(cls, config: Dict[str, Any]) -> "PluginSettings":
        """从AstrBot的配置字典创建设置"""
        return cls(
            default_collection_name=config.get("default_collection_name", "general"),
            auto_create_collection=config.get("auto_create_collection", True),
            enable_kb_llm_enhancement=config.get("enable_kb_llm_enhancement", True),
            kb_llm_search_top_k=config.get("kb_llm_search_top_k", 3),
            kb_llm_insertion_method=config.get(
                "kb_llm_insertion_method", "prepend_prompt"
            ),
            kb_llm_min_similarity_score=config.get("kb_llm_min_similarity_score", 0.5),
            kb_llm_context_template=config.get(
                "kb_llm_context_template",
                "以下是可能相关的知识库内容：\n---\n{retrieved_contexts}\n---\n请根据以上信息回答我的问题。",
            ),
            embedding=EmbeddingSettings(
                provider=config.get("embedding_provider", "openai"),
                api_url=config.get("embedding_api_url"),
                api_key=config.get("embedding_api_key"),
                model_name=config.get("embedding_model_name", "text-embedding-ada-002"),
                dimensions=config.get("embedding_dimensions"),
            ),
            llm_parser=LLMSettings(
                enable_llm_parser=config.get("enable_llm_parser", True),
                provider=config.get("llm_parser_provider"),
            ),
            rerank=RerankSettings(
                strategy=config.get("rerank_strategy", "auto"),
                api_provider=config.get("rerank_api_provider", "cohere"),
                api_key=config.get("rerank_api_key"),
                api_url=config.get("rerank_api_url"),
            ),
        )
