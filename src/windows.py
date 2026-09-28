"""Задания 36-37. Скользящее окно и скользящее среднее."""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.config import run_step, step_done
from src.io_utils import save_figure


def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    """Матрица окон формы (n - width + 1, width)."""
    if width <= 0 or width > len(x):
        raise ValueError("Некорректная ширина окна")
    n = len(x)
    idx = np.arange(width)[None, :] + np.arange(n - width + 1)[:, None]
    return x[idx]


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Скользящее среднее — среднее по каждому окну."""
    w = sliding_window(x, width)
    return w.mean(axis=1)


def run_windows(data: np.ndarray, width: int = 24,
                col_index: int = 1) -> dict:
    """Задания 36-37 на одного из столбцов data."""
    x = data[:, col_index]
    results = {}

    run_step(36, "Скользящее окно")
    w = sliding_window(x, width)
    results["windows"] = (w.shape, w[0].tolist())
    print(f"      форма окна: {w.shape}, первое окно: {np.round(w[0], 4)}")
    print(f"      (альтернатива: numpy.lib.stride_tricks.sliding_window_view)")
    step_done()

    run_step(37, "Скользящее среднее")
    ma = moving_average(x, width)
    results["ma"] = (ma.shape, float(ma[-1]), float(ma.mean()))
    print(f"      скользящее среднее: {ma.shape}, последнее = {ma[-1]:.4f}, "
          f"среднее по МА = {ma.mean():.4f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, lw=0.7, label=f"столбец {col_index}")
    ax.plot(np.arange(width - 1, len(x)), ma, lw=1.5, color="C1",
            label=f"MA({width})")
    ax.set_title("Скользящее окно и скользящее среднее")
    ax.set_xlabel("Индекс")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_figure(fig, "task36_37_windows")
    plt.close(fig)
    results["figure"] = os.path.basename(path)
    step_done(results["figure"])

    return results