"""
Centralized configuration management with validation
"""
import os
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from enum import Enum

class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"

class Settings(BaseSettings):
    # API Keys
    serpapi_key: Optional[str] = Field(None, env="SERPAPI_KEY")
    pexels_api_key: Optional[str] = Field(None, env="PEXELS_API_KEY")
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Model Configuration
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.2:3b-instruct-q4_0", env="OLLAMA_MODEL")
    groq_model_writer: str = Field("llama3-8b-8192", env="GROQ_MODEL_WRITER")
    groq_model_summary: str = Field("deepseek-r1-distill-llama-70b", env="GROQ_MODEL_SUMMARY")
    
    # Database
    database_url: str = Field("sqlite:///./blog.db", env="DATABASE_URL")
    
    # MCP Configuration
    mcp_config_file: str = Field("multiserver_setup_config.json", env="MCP_CONFIG_FILE")
    use_mcp: bool = Field(False, env="USE_MCP")
    
    # Performance Settings
    max_research_articles: int = Field(5, env="MAX_RESEARCH_ARTICLES")
    max_images: int = Field(3, env="MAX_IMAGES")
    chunk_size: int = Field(250, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    
    # Timeouts
    api_timeout: int = Field(30, env="API_TIMEOUT")
    ollama_timeout: int = Field(60, env="OLLAMA_TIMEOUT")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("logs/blog_system.log", env="LOG_FILE")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v):
        if not v.startswith(('sqlite:///', 'postgresql://', 'mysql://')):
            raise ValueError('Invalid database URL format')
        return v
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }

# Global settings instance
settings = Settings()

def get_model_config(provider: ModelProvider) -> Dict[str, Any]:
    """Get model configuration for specific provider"""
    configs = {
        ModelProvider.OLLAMA: {
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_model,
            "timeout": settings.ollama_timeout,
            "temperature": 0.0,
            "num_ctx": 1024
        },
        ModelProvider.GROQ: {
            "writer_model": settings.groq_model_writer,
            "summary_model": settings.groq_model_summary,
            "temperature": 0.2,
            "timeout": settings.api_timeout
        }
    }
    return configs.get(provider, {})
