"""Configuration models for AMAOS.

This module defines shared configuration models used throughout the AMAOS system.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration model with common settings."""
    
    # Use modern Pydantic v2 configuration
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for backward compatibility
        frozen=False,   # Allow mutation
        validate_assignment=True  # Validate on attribute assignment
    )
    
    # Common fields for all configs
    id: str
    name: str
    description: Optional[str] = ""
    enabled: bool = True
    

class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True
    

class MetricsConfig(BaseModel):
    """Metrics and monitoring configuration."""
    
    enabled: bool = True
    endpoint: Optional[str] = None
    push_interval: int = 15  # Seconds
    labels: Dict[str, str] = Field(default_factory=dict)


class RetryPolicy(BaseModel):
    """Configuration for retry policies."""
    
    max_retries: int = 3
    initial_delay: float = 0.1  # Seconds
    backoff_factor: float = 2.0
    max_delay: float = 10.0  # Seconds
    jitter: bool = True


class MemoryConfig(BaseModel):
    """Memory subsystem configuration."""
    
    storage_type: Literal["in_memory", "redis", "hybrid"] = "in_memory"
    redis_url: Optional[str] = None
    ttl_short_term: int = 3600  # 1 hour
    ttl_long_term: int = 2592000  # 30 days
    ttl_semantic: Optional[int] = None  # No expiry by default


class SystemConfig(BaseModel):
    """System-wide configuration."""
    
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    retry_policies: Dict[str, RetryPolicy] = Field(
        default_factory=lambda: {
            "default": RetryPolicy(),
            "api_calls": RetryPolicy(max_retries=5, initial_delay=0.5),
            "critical": RetryPolicy(max_retries=10, max_delay=30.0)
        }
    )
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
