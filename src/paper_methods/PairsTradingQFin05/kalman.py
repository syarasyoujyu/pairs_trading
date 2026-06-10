"""
Forward Kalman filter — equations (22)–(26) of Elliott et al. (2005).

Initialization (section 3.1): x̂_0 = y_0,  Σ_{0|0} = D²
"""
from dataclasses import dataclass
import numpy as np
from .model import ModelParams


@dataclass
class KalmanState:
    mu: np.ndarray         # x̂_{k|k}    filtered means       shape (N+1,)
    R: np.ndarray          # Σ_{k|k}    filtered variances   shape (N+1,)
    mu_pred: np.ndarray    # x̂_{k|k-1}  predicted means      shape (N+1,)
    sigma_pred: np.ndarray # Σ_{k|k-1}  predicted variances  shape (N+1,)
    K: np.ndarray          # Kalman gains K_{k}              shape (N+1,)


def kalman_filter(y: np.ndarray, params: ModelParams) -> KalmanState:
    """Run forward Kalman filter on observation series y.

    Args:
        y:      Observed spread series y_0 … y_N, shape (N+1,).
        params: Model parameters (A, B, C², D²).

    Returns:
        KalmanState with filtered and predicted quantities.
    """
    N = len(y) - 1
    A, B, C2, D2 = params.A, params.B, params.C2, params.D2

    mu = np.zeros(N + 1)
    R = np.zeros(N + 1)
    mu_pred = np.zeros(N + 1)
    sigma_pred = np.zeros(N + 1)
    K_arr = np.zeros(N + 1)

    # Initialisation
    mu[0] = y[0]
    R[0] = D2
    mu_pred[0] = y[0]
    sigma_pred[0] = D2
    K_arr[0] = 1.0  # first step Kalman gain is 1

    for k in range(N):
        # Predict step (eq. 22–23)
        mu_pred[k + 1] = A + B * mu[k]
        sigma_pred[k + 1] = B**2 * R[k] + C2

        # Update step (eq. 24–26)
        K = sigma_pred[k + 1] / (sigma_pred[k + 1] + D2)
        mu[k + 1] = mu_pred[k + 1] + K * (y[k + 1] - mu_pred[k + 1])
        R[k + 1] = D2 * K  # ≡ sigma_pred[k+1] * (1 - K)
        K_arr[k + 1] = K

    return KalmanState(mu=mu, R=R, mu_pred=mu_pred, sigma_pred=sigma_pred, K=K_arr)
