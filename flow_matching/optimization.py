from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QHOptimizationDefaults:
    """Validated defaults used by the 309-trajectory QH experiment."""

    candidate_count: int = 32
    iterations: int = 200
    directions: int = 64
    perturbation: float = 0.005
    learning_rate: float = 0.02
    beta1: float = 0.7
    beta2: float = 0.999
    flow_steps: int = 128
    gradient_mode: str = "random-orthogonal"


QH_OPTIMIZATION_DEFAULTS = QHOptimizationDefaults()


__all__ = ["QHOptimizationDefaults", "QH_OPTIMIZATION_DEFAULTS"]
