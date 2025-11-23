"""Enterprise-grade AI Agent Token Optimizer.

This package provides ML-driven token optimization for AI agents with features:
- Thompson Sampling for strategy selection
- Pareto optimization for multi-objective balancing
- Distributed execution with Ray
- Circuit breakers and self-healing
- Comprehensive monitoring and metrics
"""

from agent_token_optimizer.optimizer import TokenOptimizer
from agent_token_optimizer.strategies import OptimizationStrategy
from agent_token_optimizer.config import OptimizerConfig

__version__ = "1.0.0"
__all__ = ["TokenOptimizer", "OptimizationStrategy", "OptimizerConfig"]
