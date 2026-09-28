"""Задания 34-35. Сравнение собственных скейлеров со sklearn и инверсия."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import run_step, step_done
from src.io_utils import save_figure, write_table
from src.preprocess import MinMaxScalerCustom, StandardScalerCustom


def _save(fig, base: str) -> str:
    path = save_figure(fig, base)
    plt.close(fig)
    return os.path.basename(path)


def run_scaling(df: pd.DataFrame) -> dict:
    """Задания 34-35: первый столбец, свои скейлеры против sklearn."""
    data = df.to_numpy(dtype=float)
    x = data[:, [0]]  # первый столбец
    results = {}

    # ---------------- 34. Сравнение со sklearn ----------------
    run_step(34, "Сравнение собственного и sklearn-масштабирования")
    custom_mm = MinMaxScalerCustom().fit(x)
    sk_mm = MinMaxScaler().fit(x)
    custom_std = StandardScalerCustom().fit(x)
    sk_std = StandardScaler().fit(x)

    x_custom = custom_mm.transform(x)
    x_sk = sk_mm.transform(x)
    assert np.allclose(x_custom, x_sk, atol=1e-10), \
        "MinMax не совпал со sklearn"
    z_custom = custom_std.transform(x)
    z_sk = sk_std.transform(x)
    assert np.allclose(z_custom, z_sk, atol=1e-10), \
        "StandardScaler не совпал со sklearn"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x_custom[:, 0], lw=0.8, label="собственный")
    ax[0].plot(x_sk[:, 0], lw=0.8, ls="--", label="sklearn")
    ax[0].set_title("MinMax [0, 1]")
    ax[0].legend()
    ax[1].plot(z_custom[:, 0], lw=0.8, label="собственный")
    ax[1].plot(z_sk[:, 0], lw=0.8, ls="--", label="sklearn")
    ax[1].set_title("Standard (μ=0, σ=1)")
    ax[1].legend()
    fig.tight_layout()
    step_done(_save(fig, "task34_scalers"))

    # ---------------- 35. Инверсия ----------------
    run_step(35, "Инверсия (inverse_transform)")
    x_back_custom = custom_mm.inverse_transform(x_custom)
    x_back_sk = sk_mm.inverse_transform(x_sk)
    assert np.allclose(x, x_back_custom, atol=1e-10)
    assert np.allclose(x, x_back_sk, atol=1e-10)

    df_out = pd.DataFrame({
        "original": x[:, 0],
        "custom_minmax": x_custom[:, 0],
        "sklearn_minmax": x_sk[:, 0],
        "custom_std": z_custom[:, 0],
        "sklearn_std": z_sk[:, 0],
        "restored": x_back_custom[:, 0],
    })
    paths = write_table(df_out, "task34_35_scaled")
    results["paths"] = paths
    e_mm = float(np.max(np.abs(x - x_back_custom)))
    e_std = float(np.max(np.abs(x - custom_std.inverse_transform(z_custom))))
    step_done(f"файлы: {', '.join(paths)}; max|Δ| = {e_mm:.2e}")

    return results