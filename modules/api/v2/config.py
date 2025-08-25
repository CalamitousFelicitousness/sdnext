"""
Configuration management for API v2

This module handles v2-specific configuration settings,
separate from the main SDNext configuration.
"""

from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from pathlib import Path


class ApiV2Config(BaseSettings):
    """
    Configuration settings for API v2.
    
    These settings can be overridden via environment variables
    with the prefix "SDNEXT_V2_".
    """
    
    # API Settings
    api_host: str = Field("0.0.0.0", env="SDNEXT_V2_API_HOST")
    api_port: int = Field(8000, env="SDNEXT_V2_API_PORT")
    api_workers: int = Field(1, env="SDNEXT_V2_API_WORKERS")
    
    # Redis Settings
    redis_enabled: bool = Field(False, env="SDNEXT_V2_REDIS_ENABLED")
    redis_host: str = Field("localhost", env="SDNEXT_V2_REDIS_HOST")
    redis_port: int = Field(6379, env="SDNEXT_V2_REDIS_PORT")
    redis_db: int = Field(0, env="SDNEXT_V2_REDIS_DB")
    redis_password: Optional[str] = Field(None, env="SDNEXT_V2_REDIS_PASSWORD")
    
    # Cache Settings
    cache_enabled: bool = Field(True, env="SDNEXT_V2_CACHE_ENABLED")
    cache_ttl: int = Field(3600, env="SDNEXT_V2_CACHE_TTL")
    cache_max_size: int = Field(1000, env="SDNEXT_V2_CACHE_MAX_SIZE")
    
    # Queue Settings
    queue_max_size: int = Field(100, env="SDNEXT_V2_QUEUE_MAX_SIZE")
    queue_timeout: int = Field(300, env="SDNEXT_V2_QUEUE_TIMEOUT")
    queue_workers: int = Field(4, env="SDNEXT_V2_QUEUE_WORKERS")
    
    # WebSocket Settings
    ws_heartbeat_interval: int = Field(30, env="SDNEXT_V2_WS_HEARTBEAT")
    ws_max_connections: int = Field(1000, env="SDNEXT_V2_WS_MAX_CONNECTIONS")
    
    # Storage Settings
    storage_hot_path: Path = Field(Path("outputs/v2/hot"), env="SDNEXT_V2_STORAGE_HOT")
    storage_warm_path: Path = Field(Path("outputs/v2/warm"), env="SDNEXT_V2_STORAGE_WARM")
    storage_cold_path: Path = Field(Path("outputs/v2/cold"), env="SDNEXT_V2_STORAGE_COLD")
    
    # Security Settings
    jwt_secret: Optional[str] = Field(None, env="SDNEXT_V2_JWT_SECRET")
    jwt_algorithm: str = Field("HS256", env="SDNEXT_V2_JWT_ALGORITHM")
    jwt_expiry: int = Field(86400, env="SDNEXT_V2_JWT_EXPIRY")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(True, env="SDNEXT_V2_RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(100, env="SDNEXT_V2_RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="SDNEXT_V2_RATE_LIMIT_PERIOD")
    
    class Config:
        env_prefix = "SDNEXT_V2_"
        env_file = ".env.v2"
        case_sensitive = False


# Global configuration instance
config = ApiV2Config()
