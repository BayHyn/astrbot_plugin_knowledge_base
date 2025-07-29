from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class EmbeddingSettings(BaseModel):
    """Embedding model settings"""
    provider: str = Field("openai", description="Embedding provider, e.g., 'openai'")
    api_url: Optional[str] = Field(None, description="API URL for the embedding service")
    api_key: Optional[str] = Field(None, description="API key for the embedding service")
    model_name: str = Field("text-embedding-ada-002", description="Name of the embedding model")
    dimensions: Optional[int] = Field(None, description="Embedding dimensions, auto-detected if not set")

class LLMSettings(BaseModel):
    """LLM provider settings for file parsing"""
    enable_llm_parser: bool = Field(True, description="Enable LLM for parsing complex files like images")
    provider: Optional[str] = Field(None, description="LLM provider ID from AstrBot")
    
class RerankSettings(BaseModel):
    """Rerank configuration"""
    strategy: str = Field("auto", description="Rerank strategy: auto, api, cross_encoder, simple")
    api_provider: str = Field("cohere", description="Rerank API provider")
    api_key: Optional[str] = Field(None, description="Rerank API key")
    api_url: Optional[str] = Field(None, description="Rerank API URL")

class PluginSettings(BaseModel):
    """Main plugin settings"""
    default_collection_name: str = Field("general", description="Default collection name for new users")
    auto_create_collection: bool = Field(True, description="Automatically create a collection if it doesn't exist")
    enable_kb_llm_enhancement: bool = Field(True, description="Enable LLM request enhancement with KB content")
    kb_llm_search_top_k: int = Field(3, description="Number of documents to retrieve for LLM enhancement")
    kb_llm_insertion_method: str = Field("prepend_prompt", description="How to insert KB content: prepend_prompt or system_prompt")
    kb_llm_min_similarity_score: float = Field(0.5, description="Minimum similarity score to consider a document relevant")
    
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm_parser: LLMSettings = Field(default_factory=LLMSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)

    @classmethod
    def from_astrbot_config(cls, config: Dict[str, Any]) -> 'PluginSettings':
        """Create settings from AstrBot's config dictionary"""
        return cls(
            default_collection_name=config.get("default_collection_name", "general"),
            auto_create_collection=config.get("auto_create_collection", True),
            enable_kb_llm_enhancement=config.get("enable_kb_llm_enhancement", True),
            kb_llm_search_top_k=config.get("kb_llm_search_top_k", 3),
            kb_llm_insertion_method=config.get("kb_llm_insertion_method", "prepend_prompt"),
            kb_llm_min_similarity_score=config.get("kb_llm_min_similarity_score", 0.5),
            embedding=EmbeddingSettings(
                provider=config.get("embedding_provider", "openai"),
                api_url=config.get("embedding_api_url"),
                api_key=config.get("embedding_api_key"),
                model_name=config.get("embedding_model_name", "text-embedding-ada-002"),
                dimensions=config.get("embedding_dimensions")
            ),
            llm_parser=LLMSettings(
                enable_llm_parser=config.get("enable_llm_parser", True),
                provider=config.get("llm_parser_provider")
            ),
            rerank=RerankSettings(
                strategy=config.get("rerank_strategy", "auto"),
                api_provider=config.get("rerank_api_provider", "cohere"),
                api_key=config.get("rerank_api_key"),
                api_url=config.get("rerank_api_url")
            )
        )