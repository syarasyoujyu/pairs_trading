"""
Backward Kalman smoother — Shumway & Stoffer (1982).

Equations (34)–(38) of Elliott et al. (2005).

Runs backward from k=N to k=0 to compute smoothed estimates
x̂_{k|N} = E[x_k | Y_N] and cross-covariances Σ_{k,k+1|N}.
"""
from dataclasses import dataclass
import numpy as np
from .model import ModelParams
from .kalman import KalmanState


@dataclass
class SmootherState:
    mu_s: np.ndarray    # x̂_{k|N}      smoothed means              shape (N+1,)
    Sig_s: np.ndarray   # Σ_{k|N}      smoothed variances           shape (N+1,)
    Sig_lag: np.ndarray # Σ_{k,k+1|N}  lagged cross-covariances     shape (N,)
    J: np.ndarray       # J_k          smoother gains               shape (N,)


def kalman_smoother(kf: KalmanState, params: ModelParams) -> SmootherState:
    """Run backward Kalman smoother given forward filter output.

    Args:
        kf:     Output of kalman_filter().
        params: Model parameters (A, B, C², D²).

    Returns:
        SmootherState with smoothed means, variances, and cross-covariances.
    """
    N = len(kf.mu) - 1
    A, B, D2 = params.A, params.B, params.D2

    mu_s = np.zeros(N + 1)
    Sig_s = np.zeros(N + 1)
    J = np.zeros(N)          # J[k] = J_k,  k = 0 … N-1

    # Initialise backward recursion at k = N  (eq. 35–36)
    mu_s[N] = kf.mu[N]
    Sig_s[N] = kf.R[N]

    # Backward sweep to compute J, x̂_{k|N}, Σ_{k|N}
    for k in range(N - 1, -1, -1):
        # Smoother gain  J_k = B * Σ_{k|k} / Σ_{k+1|k}  (eq. 34)
        J[k] = B * kf.R[k] / kf.sigma_pred[k + 1]

        # x̂_{k|N} (eq. 35)
        mu_s[k] = kf.mu[k] + J[k] * (mu_s[k + 1] - (A + B * kf.mu[k]))

        # Σ_{k|N} (eq. 36)
        Sig_s[k] = kf.R[k] + J[k]**2 * (Sig_s[k + 1] - kf.sigma_pred[k + 1])

    # Lagged cross-covariances  Sig_lag[k] = Σ_{k,k+1|N},  k = 0 … N-1
    Sig_lag = np.zeros(N)

    # Initialise with eq. (38): Σ_{N-1,N|N} = B*(1 - K_N)*Σ_{N-1|N-1}
    Sig_lag[N - 1] = B * (1.0 - kf.K[N]) * kf.R[N - 1]

    # Backward recursion for eq. (37):
    # Σ_{k,k+1|N} = J_k*Σ_{k+1|k+1} + J_{k+1}*J_k*(Σ_{k+1,k+2|N} - B*Σ_{k+1|k+1})
    for k in range(N - 2, -1, -1):
        Sig_lag[k] = (
            J[k] * kf.R[k + 1]
            + J[k + 1] * J[k] * (Sig_lag[k + 1] - B * kf.R[k + 1])
        )

    return SmootherState(mu_s=mu_s, Sig_s=Sig_s, Sig_lag=Sig_lag, J=J)
