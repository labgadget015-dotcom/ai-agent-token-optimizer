"""Main TokenOptimizer implementation with ML-driven strategy selection."""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from agent_token_optimizer.config import OptimizerConfig
from agent_token_optimizer.strategies import (
    OptimizationStrategy,
    CachingStrategy,
    VerbosityStrategy,
    ContextPruningStrategy,
    QueryOptimizationStrategy,
)
from agent_token_optimizer.thompson_sampling import ThompsonSampler
from agent_token_optimizer.pareto import ParetoOptimizer
from agent_token_optimizer.monitoring import MetricsCollector, OptimizerMetrics
from agent_token_optimizer.circuit_breaker import CircuitBreaker
from agent_token_optimizer.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    success: bool
    original_tokens: int
    optimized_tokens: int
    reduction_percent: float
    strategy_used: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class TokenOptimizer:
    """Enterprise-grade AI Agent Token Optimizer.
    
    Features:
    - Thompson Sampling for intelligent strategy selection
    - Pareto optimization for multi-objective balancing
    - Circuit breakers and rate limiting
    - Comprehensive monitoring and metrics
    - Distributed execution support
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize the optimizer.
        
        Args:
            config: Optimizer configuration. Uses defaults if not provided.
        """
        self.config = config or OptimizerConfig()
        self._initialize_components()
        logger.info(f"TokenOptimizer initialized with mode: {self.config.mode}")

    def _initialize_components(self) -> None:
        """Initialize all optimizer components."""
        # Strategy registry
        self.strategies: Dict[str, OptimizationStrategy] = {
            "caching": CachingStrategy(self.config),
            "verbosity": VerbosityStrategy(self.config),
            "context_pruning": ContextPruningStrategy(self.config),
            "query_opt": QueryOptimizationStrategy(self.config),
        }

        # ML-driven strategy selection
        if self.config.thompson_sampling.enabled:
            self.strategy_selector = ThompsonSampler(
                strategies=list(self.strategies.keys()),
                config=self.config.thompson_sampling,
            )
        else:
            self.strategy_selector = None

        # Pareto optimization
        if self.config.pareto.enabled:
            self.pareto_optimizer = ParetoOptimizer(self.config.pareto)
        else:
            self.pareto_optimizer = None

        # Safety mechanisms
        if self.config.circuit_breaker.enabled:
            self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker)
        else:
            self.circuit_breaker = None

        if self.config.rate_limit.enabled:
            self.rate_limiter = RateLimiter(self.config.rate_limit)
        else:
            self.rate_limiter = None

        # Monitoring
        if self.config.monitoring.enabled:
            self.metrics = MetricsCollector(self.config.monitoring)
        else:
            self.metrics = None

        # Distributed execution
        self._init_distributed()

    def _init_distributed(self) -> None:
        """Initialize distributed execution if enabled."""
        if self.config.distributed.enabled:
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(
                        address=self.config.distributed.ray_address,
                        num_cpus=self.config.distributed.num_cpus,
                        num_gpus=self.config.distributed.num_gpus,
                    )
                logger.info("Distributed execution enabled with Ray")
            except ImportError:
                logger.warning("Ray not installed, distributed execution disabled")
                self.config.distributed.enabled = False

    def optimize(self, content: str, context: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize content to reduce token usage.
        
        Args:
            content: The content to optimize
            context: Optional context for optimization decisions
            
        Returns:
            OptimizationResult with optimization details
        """
        start_time = time.time()
        context = context or {}

        try:
            # Rate limiting check
            if self.rate_limiter and not self.rate_limiter.acquire():
                return OptimizationResult(
                    success=False,
                    original_tokens=self._estimate_tokens(content),
                    optimized_tokens=self._estimate_tokens(content),
                    reduction_percent=0.0,
                    strategy_used="none",
                    latency_ms=0.0,
                    error="Rate limit exceeded",
                )

            # Circuit breaker check
            if self.circuit_breaker and not self.circuit_breaker.allow_request():
                return OptimizationResult(
                    success=False,
                    original_tokens=self._estimate_tokens(content),
                    optimized_tokens=self._estimate_tokens(content),
                    reduction_percent=0.0,
                    strategy_used="none",
                    latency_ms=0.0,
                    error="Circuit breaker open",
                )

            # Select optimization strategy
            strategy_name = self._select_strategy(context)
            strategy = self.strategies[strategy_name]

            # Apply optimization
            original_tokens = self._estimate_tokens(content)
            optimized_content = strategy.apply(content, context)
            optimized_tokens = self._estimate_tokens(optimized_content)

            reduction = ((original_tokens - optimized_tokens) / original_tokens * 100 
                        if original_tokens > 0 else 0.0)

            latency_ms = (time.time() - start_time) * 1000

            result = OptimizationResult(
                success=True,
                original_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
                reduction_percent=reduction,
                strategy_used=strategy_name,
                latency_ms=latency_ms,
            )

            # Update strategy performance
            if self.strategy_selector:
                self.strategy_selector.update(
                    strategy_name,
                    reward=reduction / 100.0,  # Normalize to 0-1
                )

            # Record metrics
            if self.metrics:
                self.metrics.record_optimization(result)

            # Update circuit breaker on success
            if self.circuit_breaker:
                self.circuit_breaker.record_success()

            logger.debug(
                f"Optimized with {strategy_name}: "
                f"{original_tokens} -> {optimized_tokens} tokens "
                f"({reduction:.1f}% reduction)"
            )

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            
            # Update circuit breaker on failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()

            return OptimizationResult(
                success=False,
                original_tokens=self._estimate_tokens(content),
                optimized_tokens=self._estimate_tokens(content),
                reduction_percent=0.0,
                strategy_used="none",
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def optimize_async(self, content: str, context: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Async version of optimize."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.optimize, content, context)

    def _select_strategy(self, context: Dict[str, Any]) -> str:
        """Select the best optimization strategy."""
        if self.strategy_selector:
            return self.strategy_selector.select()
        
        # Fallback to mode-based selection
        mode = self.config.mode
        if mode == "aggressive":
            return "context_pruning"
        elif mode == "conservative":
            return "caching"
        else:  # balanced
            return "verbosity"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def get_metrics(self) -> Dict[str, Any]:
        """Get current optimizer metrics."""
        if self.metrics:
            return self.metrics.get_summary()
        return {}

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        if self.metrics:
            self.metrics.reset()

    def shutdown(self) -> None:
        """Shutdown the optimizer and cleanup resources."""
        if self.config.distributed.enabled:
            try:
                import ray
                ray.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}")
        
        logger.info("TokenOptimizer shut down")

    def __enter__(self) -> "TokenOptimizer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
