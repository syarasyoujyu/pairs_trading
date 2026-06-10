"""Render OHLC windows into RGB candlestick tensors."""
import numpy as np
import pandas as pd


GREEN = np.asarray([0.05, 0.55, 0.20], dtype=np.float32)
RED = np.asarray([0.85, 0.12, 0.12], dtype=np.float32)
BLACK = np.asarray([0.05, 0.05, 0.05], dtype=np.float32)
BACKGROUND = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)


def render_candlestick_window(ohlc: pd.DataFrame, image_size: int) -> np.ndarray:
    """Render one OHLC window into a 64x64 RGB-like image tensor."""
    log_ohlc = np.log(ohlc[["open", "high", "low", "close"]].astype(float).to_numpy())
    y_pixels = price_values_to_pixels(log_ohlc, image_size)
    canvas = np.ones((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = BACKGROUND
    candle_width = max(1, image_size // max(1, len(ohlc) * 2))
    for day_index, row in enumerate(y_pixels):
        x = day_index_to_pixel(day_index, len(ohlc), image_size)
        open_y, high_y, low_y, close_y = row
        color = GREEN if close_y <= open_y else RED
        draw_vertical_line(canvas, x, high_y, low_y, BLACK)
        draw_body(canvas, x, open_y, close_y, candle_width, color)
    return canvas


def price_values_to_pixels(values: np.ndarray, image_size: int) -> np.ndarray:
    """Map log OHLC values to image row positions."""
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    scale = maximum - minimum
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    normalized = (values - minimum) / scale
    pixels = (image_size - 1) - np.rint(normalized * (image_size - 1)).astype(int)
    return np.clip(pixels, 0, image_size - 1)


def day_index_to_pixel(day_index: int, day_count: int, image_size: int) -> int:
    """Map a day index to an image x-coordinate."""
    if day_count <= 1:
        return image_size // 2
    return int(round(day_index / (day_count - 1) * (image_size - 1)))


def draw_vertical_line(canvas: np.ndarray, x: int, y_a: int, y_b: int, color: np.ndarray) -> None:
    """Draw one vertical wick line."""
    top = min(y_a, y_b)
    bottom = max(y_a, y_b)
    canvas[top:bottom + 1, x, :] = color


def draw_body(canvas: np.ndarray, x: int, open_y: int, close_y: int, candle_width: int, color: np.ndarray) -> None:
    """Draw one candlestick body."""
    top = min(open_y, close_y)
    bottom = max(open_y, close_y)
    if top == bottom:
        bottom = min(canvas.shape[0] - 1, bottom + 1)
    left = max(0, x - candle_width)
    right = min(canvas.shape[1] - 1, x + candle_width)
    canvas[top:bottom + 1, left:right + 1, :] = color
