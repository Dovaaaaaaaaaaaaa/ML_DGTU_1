"""Задания 46-49. Матричные операции и пайплайн sklearn (PCA)."""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import run_step, step_done
from src.io_utils import save_figure
from src.preprocess import MinMaxScalerCustom, StandardScalerCustom


def random_sparse(m: int, n: int, density: float = 0.01,
                  format: str = "coo", rng=None) -> sparse.spmatrix:
    """Разреженная матрица размера m x n с заданной плотностью.

    Заменяет scipy.sparse.random на том же API, но без зависания на больших
    разреженных размерах.

    Почему нельзя звать scipy.sparse.random напрямую: внутри он делает
        raveled_ind = rng.choice(m * n, size=nnz, replace=False)
    а numpy для choice без замены строит ПЕРЕСТАНОВКУ всей генеральной
    совокупности. Для 100000 x 100000 это 10**10 элементов, то есть попытка
    выделить ~80 ГБ: процесс надолго уходит в подкачку памяти и вешает
    всю систему. При density = 1e-5 ненулевых элементов всего 100000,
    поэтому достаточно взять случайные линейные индексы С ПОВТОРОМ
    (O(nnz) памяти) и убрать дубликаты через np.unique. Ожидаемое число
    коллизий size**2 / (2 * m * n) = 0.5, то есть плотность сохраняется.
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError("плотность должна быть в диапазоне 0 <= d <= 1")

    total = m * n
    size = int(round(density * total))
    if size == 0:
        return sparse.coo_matrix((m, n)).asformat(format)

    # По умолчанию берём глобальный генератор numpy, а не отдельный
    # default_rng: его засевает set_seeds(), поэтому задание 46
    # воспроизводится при повторном запуске main.py.
    if rng is None:
        flat = np.unique(np.random.randint(0, total, size=size, dtype=np.int64))
        data = np.random.uniform(size=flat.size)
    else:
        flat = np.unique(rng.integers(0, total, size=size, dtype=np.int64))
        data = rng.uniform(size=flat.size)

    mat = sparse.coo_matrix(
        (data, (flat // n, flat % n)),
        shape=(m, n),
    )
    return mat.asformat(format)


def _numeric_rank(x: np.ndarray, tol: float = 1e-10) -> int:
    """Числовой ранг матрицы через сингулярные значения.

    Нужен, потому что F из задания 48 (500 x 12) заведомо вырожден:
    каждый из блоков MinMax(X) и Standard(X) — аффинное преобразование X,
    то есть линейная комбинация его столбцов ПЛЮС свободный член. Поэтому
    rank(F) = rank(X) + 1 = 5, а не 12, и PCA() по умолчанию (12 компонент)
    даёт нулевые собственные значения, из-за чего log-likelihood уезжает
    либо в -inf, либо (если зануления заменить на eps) в мусорные 1e14.

    Разрыв спектра колоссальный: s[4] = 2.9e+01, s[5] = 1.7e-14, так что
    граница ранга однозначна.
    """
    s = np.linalg.svd(x, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int(np.sum(s > s[0] * tol))


def run_matrix_ops(data: np.ndarray) -> dict:
    """Задания 46-49 на матрице X формы (n, m)."""
    X = data
    results = {}

    # ---------------- 46. Разреженный массив SciPy ----------------
    run_step(46, "Разреженный массив SciPy (100000×100000)")
    S = random_sparse(100_000, 100_000, density=1e-5, format="csr")
    results["nnz"] = int(S.nnz)
    results["shape"] = S.shape
    results["density"] = S.nnz / (S.shape[0] * S.shape[1])
    step_done(f"NNZ = {S.nnz}, плотность = {results['density']:.2e}, "
              f"память ≈ {(S.data.nbytes + S.indices.nbytes + S.indptr.nbytes) / 1e6:.1f} МБ")

    # ---------------- 47. Обратная матрица ----------------
    run_step(47, "Обратная матрица 100×100")
    M = np.random.rand(100, 100)
    det = float(np.linalg.det(M))
    if abs(det) > 1e-10:
        M_inv = np.linalg.inv(M)
        err = float(np.max(np.abs(M @ M_inv - np.eye(100))))
        assert err < 1e-6
        results["det"] = det
        step_done(f"det = {det:.4f}, max|M·M⁻¹ - I| = {err:.2e}")
    else:
        print("      матрица вырождена")
        step_done("det ≈ 0")

    # ---------------- 48. Достроение до F (n, m, 3) ----------------
    run_step(48, "Достроение массива до F (n, m, 3)")
    X_mm = MinMaxScalerCustom().fit_transform(X)
    X_std = StandardScalerCustom().fit_transform(X)
    F = np.stack([X, X_mm, X_std], axis=-1)
    results["F"] = F.shape
    print(f"      F {F.shape}")
    step_done()

    # ---------------- 49. Пайплайн sklearn (PCA) ----------------
    run_step(49, "Пайплайн sklearn (PCA), лог-правдоподобие")
    flat = F.reshape(len(F), -1)          # (n, m·3)
    rank = _numeric_rank(flat)
    results["rank"] = rank
    print(f"      F развёрнут до {flat.shape}; числовой ранг = {rank} "
          f"из {flat.shape[1]} (остальные столбцы линейно зависимы)")

    # PCA ровно на ранг: тогда все собственные значения положительны и
    # log-likelihood конечен, в отличие от PCA() по умолчанию.
    scalers = {"minmax": MinMaxScaler(), "std": StandardScaler()}
    loglik = {}
    for name, scaler in scalers.items():
        pipe = Pipeline([("scaler", scaler), ("pca", PCA(n_components=rank))])
        pipe.fit(flat)
        t = pipe.named_steps["scaler"].transform(flat)
        ll = float(pipe.named_steps["pca"].score_samples(t).sum())
        loglik[name] = ll
        print(f"      {name}: log-likelihood = {ll:.2f}")
    best = max(loglik, key=loglik.get)
    results["loglik"] = loglik
    results["best"] = best
    print(f"      лучший вариант: {best}")

    # Кумулятивная объяснённая дисперсия по всем 12 направлениям
    pca = PCA().fit(flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    results["cumvar"] = cumvar
    # хвост после rank-й компоненты нулевой, log-scale его не рисует
    fig, ax = plt.subplots()
    ax.plot(cumvar[:rank], marker="o", color="C0", label="ненулевые")
    ax.plot(cumvar[rank - 1:], marker="x", color="C3",
            ls="--", label="нулевые (зависимые)")
    ax.set_yscale("log")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Кумулятивная объяснённая дисперсия (log)")
    ax.set_title(f"PCA по F ({flat.shape[0]}x{flat.shape[1]}, ранг {rank})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = save_figure(fig, "task49_pca_cumvar")
    plt.close(fig)
    step_done(os.path.basename(path))

    return results