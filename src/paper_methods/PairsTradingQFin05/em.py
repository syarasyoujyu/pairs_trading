"""
EM parameter estimation — Shumway & Stoffer (1982) smoother approach.

Section 3.2.1 of Elliott et al. (2005).

M-step update equations (39)–(42):
    α = Σ_{k=1}^N E[x_{k-1}² | Y_N]
    β = Σ_{k=1}^N E[x_{k-1} x_k | Y_N]
    γ = Σ_{k=1}^N x̂_{k|N}
    δ = Σ_{k=1}^N x̂_{k-1|N}

    Â  = (αγ - βδ) / (Nα - δ²)
    B̂  = (Nβ - γδ) / (Nα - δ²)
    Ĉ² = (1/N)   Σ E[(x_k - Â - B̂ x_{k-1})² | Y_N]
    D̂² = (1/N+1) Σ E[(y_k - x_k)²          | Y_N]
"""
import numpy as np
from .model import ModelParams
from .kalman import kalman_filter
from .smoother import kalman_smoother


def _mstep(
    y: np.ndarray,
    mu_s: np.ndarray,
    Sig_s: np.ndarray,
    Sig_lag: np.ndarray,
) -> ModelParams:
    """Shumway-Stoffer M-step: compute updated (Â, B̂, Ĉ², D̂²)."""
    N = len(y) - 1

    # Sufficient statistics
    alpha = float(np.sum(Sig_s[:-1] + mu_s[:-1] ** 2))       # Σ E[x_{k-1}²]
    beta  = float(np.sum(Sig_lag   + mu_s[:-1] * mu_s[1:]))  # Σ E[x_{k-1} x_k]
    gamma = float(np.sum(mu_s[1:]))                            # Σ x̂_{k|N}
    delta = float(np.sum(mu_s[:-1]))                           # Σ x̂_{k-1|N}

    # eq. (39)–(40)
    denom = N * alpha - delta**2
    if abs(denom) < 1e-12:
        # Degenerate — return current-ish safe values
        A_new = max(float(np.std(y) * 0.1), 1e-6)
        B_new = 0.85
    else:
        A_new = (alpha * gamma - beta * delta) / denom
        B_new = (N * beta - gamma * delta) / denom

    A_new = max(float(A_new), 1e-8)
    B_new = float(np.clip(B_new, 1e-4, 1.0 - 1e-4))

    # eq. (41) — expanded form for Ĉ²
    C2_new = (1.0 / N) * (
        float(np.sum(Sig_s[1:] + mu_s[1:] ** 2))   # Σ E[x_k²]
        + N * A_new**2
        + B_new**2 * alpha                           # B̂² Σ E[x_{k-1}²]
        - 2.0 * A_new * gamma                        # -2Â γ
        + 2.0 * A_new * B_new * delta               # +2Â B̂ δ
        - 2.0 * B_new * beta                         # -2B̂ β
    )

    # eq. (42)
    D2_new = (1.0 / (N + 1)) * float(
        np.sum(y**2 - 2.0 * y * mu_s + Sig_s + mu_s**2)
    )

    return ModelParams(
        A=A_new,
        B=B_new,
        C2=max(float(C2_new), 1e-8),
        D2=max(float(D2_new), 1e-8),
    )


def _default_init(y: np.ndarray) -> ModelParams:
    std = float(np.std(y))
    return ModelParams(
        A=max(std * 0.1, 1e-4),
        B=0.85,
        C2=max(float(np.var(y)) * 0.25, 1e-6),
        D2=max(float(np.var(y)) * 0.40, 1e-6),
    )


def em_estimate(
    y: np.ndarray,
    init: ModelParams | None = None,
    n_iter: int = 150,
) -> ModelParams:
    """Full EM estimation on observation series y.

    Iterates the E-step (Kalman filter + smoother) and M-step `n_iter` times.
    Returns the converged ModelParams.

    Section 3.2.1 and 3.3 of Elliott et al. (2005).
    """
    params = init if init is not None else _default_init(y)

    for _ in range(n_iter):
        kf = kalman_filter(y, params)
        ks = kalman_smoother(kf, params)
        params = _mstep(y, ks.mu_s, ks.Sig_s, ks.Sig_lag)

    return params


def em_update_one(y: np.ndarray, params: ModelParams) -> ModelParams:
    """Single EM iteration for online rolling updates (section 3.3)."""
    kf = kalman_filter(y, params)
    ks = kalman_smoother(kf, params)
    return _mstep(y, ks.mu_s, ks.Sig_s, ks.Sig_lag)
