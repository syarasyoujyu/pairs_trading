"""
State-space model parameters for Elliott et al. (2005).

State equation (eq. 7):  x_{k+1} = A + B*x_k + C*ε_{k+1}
Observation equation (eq. 9): y_k = x_k + D*ω_k

Mean-reversion condition: 0 < B < 1 (i.e. 0 < 1-bτ < 1)
Long-run mean: μ∞ = A / (1 - B)
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelParams:
    A: float   # drift; must be > 0 for valid mean-reverting model
    B: float   # persistence; must be in (0, 1)
    C2: float  # state noise variance (C = σ√τ)
    D2: float  # observation noise variance

    @property
    def C(self) -> float:
        return float(np.sqrt(max(self.C2, 0.0)))

    @property
    def D(self) -> float:
        return float(np.sqrt(max(self.D2, 0.0)))

    @property
    def mu_inf(self) -> float:
        """Long-run equilibrium a/b = A / (1 - B)."""
        return self.A / (1.0 - self.B)

    def is_valid(self) -> bool:
        """Check eq. 7 validity conditions: A > 0, 0 < B < 1."""
        return self.A > 0 and 0.0 < self.B < 1.0 and self.C2 > 0 and self.D2 > 0

    def __repr__(self) -> str:
        return (
            f"ModelParams(A={self.A:.5f}, B={self.B:.5f}, "
            f"C={self.C:.5f}, D={self.D:.5f}, μ∞={self.mu_inf:.5f})"
        )
