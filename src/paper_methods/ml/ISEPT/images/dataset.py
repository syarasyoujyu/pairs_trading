"""Build monthly candlestick image datasets."""
import numpy as np
import pandas as pd

from common.config import ImageConfig
from images.candlestick import render_candlestick_window


def build_monthly_image_payload(
    panel: dict[str, pd.DataFrame],
    month_ends: list[pd.Timestamp],
    config: ImageConfig,
) -> tuple[np.ndarray, list[dict]]:
    """Build a flat image tensor and metadata rows for Modal training."""
    images: list[np.ndarray] = []
    metadata: list[dict] = []
    for month_end in month_ends:
        for symbol, frame in panel.items():
            symbol_images = candlestick_images_for_symbol_month(frame, month_end, config)
            for image_number, image in enumerate(symbol_images):
                images.append(image)
                metadata.append(
                    {
                        "month_end": month_end.strftime("%Y-%m-%d"),
                        "symbol": symbol,
                        "image_number": image_number,
                    }
                )
    if not images:
        raise ValueError("No candlestick images were generated.")
    return np.asarray(images, dtype=np.float32), metadata


def candlestick_images_for_symbol_month(frame: pd.DataFrame, month_end: pd.Timestamp, config: ImageConfig) -> list[np.ndarray]:
    """Return all 21-day images in the lookback period for one symbol-month."""
    end_position = int(frame.index.get_loc(month_end))
    start_position = max(0, end_position - config.lookback_bars + 1)
    lookback = frame.iloc[start_position:end_position + 1]
    rows = []
    for start in range(0, len(lookback) - config.candle_window + 1, config.window_step):
        window = lookback.iloc[start:start + config.candle_window]
        rows.append(render_candlestick_window(window, config.image_size))
    return rows


def available_symbols_by_month(metadata: list[dict]) -> dict[str, list[str]]:
    """Return sorted symbols available for every month in image metadata."""
    frame = pd.DataFrame(metadata)
    groups = frame.groupby("month_end")["symbol"].unique()
    return {month: sorted(symbols.astype(str).tolist()) for month, symbols in groups.items()}
