"""Configuration management for the Token Optimizer."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class OptimizationMode(str, Enum):
    """Optimization modes."""
    AGGRESSIVE = "aggressive"  # Maximum token reduction
    BALANCED = "balanced"      # Balance between tokens and quality
    CONSERVATIVE = "conservative"  # Prioritize quality over tokens


class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    max_size_mb: int = Field(default=100, ge=1, le=10000)
    ttl_seconds: int = Field(default=3600, ge=60)
    eviction_policy: str = Field(default="lru")


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    enabled: bool = True
    failure_threshold: int = Field(default=5, ge=1)
    success_threshold: int = Field(default=2, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)
    half_open_max_calls: int = Field(default=1, ge=1)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = True
    max_requests_per_minute: int = Field(default=60, ge=1)
    max_tokens_per_minute: int = Field(default=100000, ge=100)
    burst_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""
    enabled: bool = True
    export_interval_seconds: int = Field(default=60, ge=10)
    otlp_endpoint: Optional[str] = None
    log_level: str = Field(default="INFO")
    trace_sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0)


class ThompsonSamplingConfig(BaseModel):
    """Thompson Sampling strategy selection configuration."""
    enabled: bool = True
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    min_samples_per_strategy: int = Field(default=10, ge=1)
    update_interval_seconds: int = Field(default=300, ge=60)


class ParetoConfig(BaseModel):
    """Pareto optimization configuration."""
    enabled: bool = True
    objectives: List[str] = Field(default=["token_reduction", "latency", "accuracy"])
    weights: Optional[Dict[str, float]] = None
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is not None:
            total = sum(v.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


class DistributedConfig(BaseModel):
    """Distributed execution configuration."""
    enabled: bool = False
    ray_address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = 0
    object_store_memory_gb: Optional[int] = None


class OptimizerConfig(BaseModel):
    """Main optimizer configuration."""
    mode: OptimizationMode = OptimizationMode.BALANCED
    target_reduction_percent: float = Field(default=80.0, ge=0.0, le=99.0)
    max_token_budget: Optional[int] = Field(default=None, ge=100)
    
    # Sub-configurations
    cache: CacheConfig = Field(default_factory=CacheConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    thompson_sampling: ThompsonSamplingConfig = Field(default_factory=ThompsonSamplingConfig)
    pareto: ParetoConfig = Field(default_factory=ParetoConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    
    # Strategy-specific settings
    strategy_config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic config."""
        use_enum_values = True
        validate_assignment = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "OptimizerConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
