"""Задания 1-3. Чтение таблиц, запись во все форматы, сохранение фигур."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat

from src.config import timestamped_filename


def read_table(path: str) -> pd.DataFrame:
    """Универсальное чтение табличных данных по расширению файла."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".xlsx":
        df = pd.read_excel(p)
    elif ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".txt":
        # разделитель — пробел ИЛИ табуляция (методичка, задание 1)
        df = pd.read_csv(p, sep=r"\s+", engine="python")
    elif ext == ".mat":
        df = _read_mat(p)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    # Автоматическое именование столбцов, если заголовков не было
    if all(isinstance(c, int) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


def _read_mat(path: Path) -> pd.DataFrame:
    """Разбор .mat, записанного savemat(data=..., columns=...).

    В словаре помимо данных лежат служебные ключи (__header__ и т.п.),
    поэтому берём самый большой двумерный числовой массив, а имена столбцов
    восстанавливаем из массива columns.
    """
    mat = loadmat(path)
    data: np.ndarray | None = None
    cols = None

    for k, v in mat.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v)
        if a.ndim == 2 and a.size > 0 and np.issubdtype(a.dtype, np.number):
            if data is None or a.size > data.size:
                data = a
        elif k.lower() == "columns":
            # savemat кладёт одномерный массив как вектор-столбец (n, 1),
            # а каждое имя — ещё одним вложенным массивом (1,), поэтому
            # берём первый элемент каждого плоского элемента.
            cols = []
            for x in np.asarray(v).ravel():
                a = np.asarray(x).reshape(-1)
                cols.append(str(a[0]) if a.size else str(x))

    if data is None:
        raise ValueError(f"В .mat-файле {path} не найден числовой массив")

    df = pd.DataFrame(data)
    if cols and len(cols) == df.shape[1]:
        df.columns = cols
    return df


def write_table(df: pd.DataFrame, base_name: str) -> dict:
    """Сохраняет DataFrame во все форматы с timestamp в имени файла."""
    paths = {}

    paths["xlsx"] = timestamped_filename(base_name, "xlsx")
    df.to_excel(paths["xlsx"], index=False, engine="openpyxl")

    paths["csv"] = timestamped_filename(base_name, "csv")
    df.to_csv(paths["csv"], index=False)

    paths["txt"] = timestamped_filename(base_name, "txt")
    df.to_csv(paths["txt"], sep="\t", index=False)

    paths["mat"] = timestamped_filename(base_name, "mat")
    # savemat умеет сохранять только числа, поэтому на вход df обязан быть
    # числовым (в main.py для этого вызывается _clean_features).
    from scipy.io import savemat  # ленивый импорт — пакет scipy и так в памяти
    savemat(paths["mat"], {"data": df.to_numpy(),
                           "columns": np.array(df.columns, dtype=object)})
    return paths


def save_figure(fig, base_name: str) -> str:
    """Сохраняет matplotlib-фигуру в PNG и возвращает путь к файлу."""
    path = timestamped_filename(base_name, "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path