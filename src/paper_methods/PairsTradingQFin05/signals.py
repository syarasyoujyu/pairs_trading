"""
Trading signal generation — section 2.3 of Elliott et al. (2005).

At each step k the Kalman filter provides the one-step-ahead prediction:
    x̂_{k|k-1}  and  Σ_{k|k-1}

The innovation (prediction error) is:
    ν_k = y_k - x̂_{k|k-1}

Its standard deviation is:
    s_k = √(Σ_{k|k-1} + D²)

The z-score  z_k = ν_k / s_k  drives the signal:
    z_k >  c  →  LONG  spread  (+1)   spread is above predicted mean
    z_k < -c  →  SHORT spread  (-1)   spread is below predicted mean
    |z_k| ≤ c  →  FLAT          (0)
"""
import numpy as np
import pandas as pd
from .model import ModelParams
from .kalman import kalman_filter
from .em import em_estimate, em_update_one


def generate_signals(
    spread: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    window: int = 100,
    threshold: float = 1.0,
    n_em_iter: int = 150,
    reestimate_every: int = 20,
) -> tuple[pd.DataFrame, ModelParams]:
    """Generate per-bar trading signals.

    Rolling-window implementation (section 3.3):
    1. Estimate (A, B, C², D²) via full EM on the first `window` bars.
    2. At each subsequent bar k:
       a. Run Kalman filter on the last `window` bars → filtered state.
       b. Predict y_k: x̂_{k|k-1} = A + B * x̂_{k-1|k-1}.
       c. Compute z-score of the innovation.
       d. Emit signal ∈ {-1, 0, +1}.
    3. Every `reestimate_every` steps, update params with one EM iteration.

    Args:
        spread:            Observed spread series y_0 … y_T.
        dates:             Datetime index aligned with spread.
        window:            Rolling estimation window N.
        threshold:         Entry threshold c (in innovation-std units).
        n_em_iter:         EM iterations for initial estimation.
        reestimate_every:  Frequency of online parameter update.

    Returns:
        (signals DataFrame, final ModelParams)
    """
    if len(spread) < window + 1:
        raise ValueError(f"Need ≥ {window + 1} observations, got {len(spread)}.")

    # --- Initial estimation on the first window ---
    params = em_estimate(spread[:window], n_iter=n_em_iter)
    print(f"[EM init] {params}")

    if not params.is_valid():
        print("[WARN] Initial params invalid (B not in (0,1) or A≤0). "
              "Mean-reversion assumption may not hold for this spread.")

    rows: list[dict] = []
    steps_since_update = 0

    for k in range(window, len(spread)):
        y_window = spread[k - window : k]   # y_{k-window} … y_{k-1}

        # Periodic online update (section 3.3)
        if steps_since_update >= reestimate_every:
            params = em_update_one(y_window, params)
            steps_since_update = 0
        steps_since_update += 1

        # Forward Kalman on the window
        kf = kalman_filter(y_window, params)

        # One-step-ahead prediction for y[k]
        x_hat = params.A + params.B * kf.mu[-1]          # x̂_{k|k-1}
        sig_pred = params.B**2 * kf.R[-1] + params.C2    # Σ_{k|k-1}
        sig_innov = float(np.sqrt(max(sig_pred + params.D2, 1e-12)))

        y_k = float(spread[k])
        innovation = y_k - x_hat
        z = innovation / sig_innov

        if z > threshold:
            signal = 1
        elif z < -threshold:
            signal = -1
        else:
            signal = 0

        rows.append({
            "datetime":   dates[k],
            "spread":     round(y_k, 8),
            "x_pred":     round(float(x_hat), 8),
            "sigma_pred": round(float(sig_pred), 8),
            "innovation": round(float(innovation), 8),
            "z_score":    round(float(z), 6),
            "signal":     signal,
            "lots":       1.0 if signal != 0 else 0.0,
        })

    signals_df = pd.DataFrame(rows)
    return signals_df, params
