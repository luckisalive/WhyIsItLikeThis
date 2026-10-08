"""
Central application settings using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # Neo4j AuraDB
    neo4j_uri: str = Field(default="NEO4J_URI", description="Neo4j AuraDB connection URI (neo4j+s://...)")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="NEO4J_PASSWORD", description="Neo4j password")

    # GitHub
    github_token: str = Field(default="GITHUB_TOKEN", description="GitHub Personal Access Token")

    # Groq Cloud (extraction LLM)
    groq_api_key: str = Field(default="GROQ_API_KEY", description="Groq Cloud API key")

    # Google Gemini (reasoning LLM + embeddings)
    google_api_key: str = Field(default="GOOGLE_API_KEY", description="Google Gemini API key")

    # Target Repository
    target_repo: str = Field(
        default="tiangolo/fastapi",
        description="GitHub repository to analyze (owner/name)",
    )

    # Ingestion controls
    max_commits: int = Field(default=500, description="Max commits to ingest")
    max_prs: int = Field(default=300, description="Max pull requests to ingest")
    max_issues: int = Field(default=200, description="Max issues to ingest")
    batch_size: int = Field(default=10, description="LLM extraction batch size")

    # Embedding (Gemini Embedding 2 at 768 dimensions)
    embedding_model: str = Field(
        default="gemini-embedding-2",
        description="Gemini embedding model name (e.g. gemini-embedding-2)",
    )
    embedding_dimensions: int = Field(default=768, description="Embedding vector dimensions")

    # Groq reasoning / answering model
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq model for final reasoning and query answering (openai/gpt-oss-120b)",
    )
    groq_fallback_models: str = Field(
        default="openai/gpt-oss-20b",
        description="Comma-separated Groq fallback models (e.g. openai/gpt-oss-20b)",
    )

    # Gemini extraction model
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Primary Gemini model for decision extraction (gemini-3.5-flash-lite)",
    )
    gemini_fallback_models: str = Field(
        default="gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-2.5-flash,gemini-3-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash",
        description="Comma-separated list of Gemini models to cascade through upon reaching daily rate limits",
    )

    # Cross-provider fallback
    enable_cross_provider_fallback: bool = Field(
        default=True,
        description="Enable cross-provider failover (Gemini -> Groq or Groq -> Gemini) when rate limits are exhausted",
    )

    # Global rate limit cooldown when all models are exhausted
    rate_limit_cooldown_seconds: int = Field(
        default=60,
        description="Cooldown wait time in seconds if all models across both providers are exhausted",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}




def get_settings() -> Settings:
    """Factory to create and cache settings instance."""
    return Settings()
