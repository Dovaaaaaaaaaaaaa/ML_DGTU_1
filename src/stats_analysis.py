"""Задания 14-33. Статистический анализ данных средствами NumPy, Pandas, SciPy.

Модуль организован как последовательный сценарий run_statistics(), который
выполняет все двадцать заданий методички и печатает/сохраняет результаты.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import spectrogram, periodogram, convolve
from scipy.interpolate import interp1d, UnivariateSpline

from src.config import run_step, step_done
from src.io_utils import save_figure


# ----------------------------------------------------------------------
# Вспомогательные функции-«заготовки» из методички (задания 14-33)
# ----------------------------------------------------------------------
def plot_series(data: np.ndarray, title: str = "Исходные данные"):
    """Задание 14. График всех столбцов."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(data.shape[1]):
        ax.plot(data[:, i], lw=0.8, label=f"col_{i}")
    ax.set_title(title)
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_histogram(x: np.ndarray, bins: int = 30):
    """Задание 15. Гистограмма с оценкой плотности."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, density=True, alpha=0.7, edgecolor="black",
            label="гистограмма")
    if x.std() > 0:
        from scipy.stats import gaussian_kde
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, gaussian_kde(x)(xs), label="KDE", color="C1")
        ax.legend()
    ax.set_title("Нормализованная гистограмма")
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    return fig


def show_sorted(df: pd.DataFrame) -> None:
    """Задание 16. Отсортированные столбцы."""
    for col in df.columns:
        v = np.sort(df[col].to_numpy())
        head = " ".join(f"{x:.3f}" for x in v[:8])
        tail = " ".join(f"{x:.3f}" for x in v[-3:])
        print(f"      {col}: {head} ... {tail}  (всего {len(v)})")


def plot_ecdf(x: np.ndarray):
    """Задание 17. Эмпирическая функция распределения."""
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots()
    ax.step(xs, ys, where="post")
    ax.set_title("Эмпирическая функция распределения")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    return fig


def column_stats(x: np.ndarray) -> dict:
    """Задание 18. mean/var/mode/median."""
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=0)),
        "mode": float(stats.mode(x, keepdims=False).mode),
        "median": float(np.median(x)),
    }


def ci_mean(x: np.ndarray, alpha: float = 0.05):
    """Задание 19 (среднее, σ неизвестна): x ± t_{1-α/2, n-1} · s/√n."""
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)  # несмещённое СКО
    se = s / np.sqrt(n)
    return stats.t.interval(1 - alpha, df=n - 1, loc=m, scale=se)


def ci_var(x: np.ndarray, alpha: float = 0.05):
    """Задание 19 (дисперсия): ((n-1)s²/χ²_{1-α/2}, (n-1)s²/χ²_{α/2})."""
    n = len(x)
    s2 = np.var(x, ddof=1)
    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    return (n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low


def cross_correlation(x, y, max_lags: int = 50):
    """Задание 22. Нормированная взаимная корреляция по диапазону лагов."""
    lags = np.arange(-max_lags, max_lags + 1)
    c = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    c = c / (np.std(x) * np.std(y) * len(x))
    mid = len(c) // 2
    return lags, c[mid - max_lags: mid + max_lags + 1]


def plot_fft_amplitude(x):
    """Задание 30. Амплитудно-частотная характеристика через FFT."""
    n = len(x)
    fft = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0)
    half = n // 2
    fig, ax = plt.subplots()
    ax.plot(freq[:half], np.abs(fft[:half]))
    ax.set_xlabel("Частота")
    ax.set_ylabel("|X(f)|")
    return fig


def _save(fig, base: str) -> str:
    """Сохранить фигуру, закрыть её, вернуть имя файла."""
    path = save_figure(fig, base)
    plt.close(fig)
    return os.path.basename(path)


# ----------------------------------------------------------------------
# Сценарий заданий 14-33
# ----------------------------------------------------------------------
def run_statistics(df: pd.DataFrame) -> dict:
    """Последовательно выполняет задания 14-33 и возвращает результаты."""
    data = df.to_numpy(dtype=float)
    cols = list(df.columns)
    keys = {"14": None, "18": {}, "19": {}, "21": None, "22": None,
            "27": {}, "32": None}

    # ---------------- 14. График исходных данных ----------------
    run_step(14, "График исходных данных")
    fig = plot_series(data)
    keys["14"] = _save(fig, "task14_series")
    step_done(keys["14"])

    # ---------------- 15. Гистограмма и плотность ----------------
    run_step(15, "Гистограмма и плотность")
    fig = plot_histogram(data[:, 0], bins=30)
    step_done(_save(fig, "task15_hist"))

    # ---------------- 16. Отсортированные столбцы ----------------
    run_step(16, "Отсортированные столбцы")
    show_sorted(df)
    step_done()

    # ---------------- 17. Эмпирическая функция распределения ----------------
    run_step(17, "Эмпирическая функция распределения")
    fig = plot_ecdf(data[:, 0])
    step_done(_save(fig, "task17_ecdf"))

    # ---------------- 18. Статистики ----------------
    run_step(18, "Статистики (numpy + pandas)")
    for i, c in enumerate(cols):
        s = column_stats(data[:, i])
        keys["18"][c] = s
        print(f"      {c}: mean={s['mean']:.4f} var={s['var']:.4f} "
              f"mode={s['mode']:.4f} median={s['median']:.4f}")
    desc = df.describe().round(4)
    print(f"      pandas describe():\n{desc.to_string()}")
    step_done()

    # ---------------- 19. Доверительные интервалы ----------------
    run_step(19, "Доверительные интервалы (mean, var)")
    for i, c in enumerate(cols):
        keys["19"][c] = {"ci_mean": ci_mean(data[:, i]),
                         "ci_var": ci_var(data[:, i])}
        lo, hi = keys["19"][c]["ci_mean"]
        vlo, vhi = keys["19"][c]["ci_var"]
        print(f"      {c}: μ ∈ [{lo:.4f}, {hi:.4f}], σ² ∈ [{vlo:.4f}, {vhi:.4f}]")
    step_done()

    # ---------------- 20. Ковариация и корреляция ----------------
    run_step(20, "Ковариация и корреляция")
    cov, corr = np.cov(data, rowvar=False), np.corrcoef(data, rowvar=False)
    cov_df, corr_df = pd.DataFrame(data).cov(), pd.DataFrame(data).corr()
    print("      numpy cov:\n" + pd.DataFrame(cov, index=cols,
                                              columns=cols).round(4).to_string())
    print("      numpy corr:\n" + pd.DataFrame(corr, index=cols,
                                               columns=cols).round(4).to_string())
    step_done()

    # ---------------- 21. Значимость коэффициента корреляции ----------------
    run_step(21, "Значимость коэффициента корреляции")
    r, p_value = stats.pearsonr(data[:, 0], data[:, -1])
    keys["21"] = (r, p_value)
    print(f"      pearsonr({cols[0]}, {cols[-1]}): r={r:.4f}, p={p_value:.4e} -> "
          f"{'значима' if p_value < 0.05 else 'незначима'}")
    step_done()

    # ---------------- 22. Взаимная корреляция ----------------
    run_step(22, "Взаимная корреляция")
    lags, c = cross_correlation(data[:, 0], data[:, 1], max_lags=20)
    best = int(lags[np.argmax(np.abs(c))])
    keys["22"] = best
    print(f"      лаг макс. |корр.| = {best} (corr = {c[np.argmax(np.abs(c))]:.3f})")
    step_done()

    # ---------------- 23. Производная и градиент ----------------
    run_step(23, "Производная и градиент")
    dx = np.gradient(data[:, 0])
    dx2 = np.gradient(data, axis=0)
    print(f"      ndimage градиент (1D): {dx.shape}, (2D по строкам): {dx2.shape}")
    step_done()

    # ---------------- 24. Свёртка ----------------
    run_step(24, "Свёртка (numpy и scipy)")
    conv = np.convolve(data[:, 0], data[:, 1], mode="full")
    conv2 = convolve(data[:, 0], data[:, 1], mode="full")
    assert np.allclose(conv, conv2)
    step_done(f"длина {len(conv)}, np и scipy совпадают")

    # ---------------- 25. Скалярное и векторное произведения ----------------
    run_step(25, "Скалярное и векторное произведения")
    dot = np.dot(data[:, 0], data[:, 1])
    cross = np.cross(data[:3, 0], data[:3, 1])
    print(f"      dot = {dot:.4f}, cross (первые 3) = {np.round(cross, 4)}")
    step_done()

    # ---------------- 26. Нормы L1 и L2 ----------------
    run_step(26, "Нормы L1 и L2")
    l1 = np.linalg.norm(data[:, 0], ord=1)
    l2 = np.linalg.norm(data[:, 0], ord=2)
    step_done(f"||x‖₁ = {l1:.4f}, ‖x‖₂ = {l2:.4f}")

    # ---------------- 27. Проверка гипотез о распределении ----------------
    run_step(27, "Проверка гипотез о распределении")
    x1, x2 = data[:, 0], data[:, 1]
    # scipy >= 1.15 не принимает строковые имена распределений в kstest —
    # передаём явно functions: cdf.
    ks_u = stats.kstest(x1, stats.uniform.cdf,
                        args=(x1.min(), x1.max() - x1.min()))
    ks_n = stats.kstest(x2, stats.norm.cdf,
                        args=(x2.mean(), x2.std()))
    sh_n = stats.shapiro(x2)
    keys["27"] = {"ks_uniform": ks_u.pvalue, "ks_norm": ks_n.pvalue,
                  "shapiro": sh_n.pvalue}
    print(f"      {cols[0]}: KS-uniform p = {ks_u.pvalue:.4e}")
    print(f"      {cols[1]}: KS-norm  p = {ks_n.pvalue:.4e}")
    print(f"      {cols[1]}: Shapiro  p = {sh_n.pvalue:.4e}")
    step_done()

    # ---------------- 28. Спектрограмма ----------------
    run_step(28, "Спектрограмма")
    f, t, Sxx = spectrogram(data[:, 0], fs=1.0)
    fig, ax = plt.subplots()
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    ax.set_xlabel("Время")
    ax.set_ylabel("Частота")
    ax.set_title("Спектрограмма")
    step_done(_save(fig, "task28_spectrogram"))

    # ---------------- 29. Периодограмма ----------------
    run_step(29, "Периодограмма")
    f, Pxx = periodogram(data[:, 0], fs=1.0)
    fig, ax = plt.subplots()
    ax.semilogy(f, Pxx)
    ax.set_xlabel("Частота")
    ax.set_ylabel("PSD")
    ax.set_title("Периодограмма")
    step_done(_save(fig, "task29_periodogram"))

    # ---------------- 30. АЧХ через FFT ----------------
    run_step(30, "АЧХ через FFT")
    fig = plot_fft_amplitude(data[:, 0])
    step_done(_save(fig, "task30_fft"))

    # ---------------- 31. Кубическая интерполяция ----------------
    run_step(31, "Кубическая интерполяция (interp1d)")
    idx = np.arange(len(data))
    f_cubic = interp1d(idx, data[:, 0], kind="cubic", fill_value="extrapolate")
    x_new = np.linspace(0, len(data) - 1, 10 * len(data))
    y_new = f_cubic(x_new)
    step_done(f"{len(y_new)} точек, NaN нет: {not np.isnan(y_new).any()}")

    # ---------------- 32. Интерполяция сплайнами ----------------
    run_step(32, "Интерполяция сплайнами (UnivariateSpline)")
    spl = UnivariateSpline(idx, data[:, 0], k=3, s=0)
    y_spl = spl(x_new)
    keys["32"] = {"points": len(y_spl),
                  "max_dev": float(np.max(np.abs(y_spl - y_new))) if len(y_new) else 0.0}
    step_done(f"{len(y_spl)} точек, max|интерп1d - сплайн| = {keys['32']['max_dev']:.2e}")

    # ---------------- 33. Бинарные маски ----------------
    run_step(33, "Бинарные маски")
    x = data[:, 0]
    masks = {
        "x > 0": (x > 0).sum(),
        "x < 0": (x < 0).sum(),
        "x == 0": (x == 0).sum(),
        "-1 <= x <= 1": ((x >= -1) & (x <= 1)).sum(),
    }
    print("      " + ", ".join(f"{k}: {v}" for k, v in masks.items()))
    step_done()

    return keys