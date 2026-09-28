"""Задания 8-13. Предобработка: пропуски, приведение к NumPy, масштабирование, разбиение."""

import numpy as np
import pandas as pd

from src.data_loader import check_data_quality


def handle_missing(df: pd.DataFrame, col: str, action: str, value=None) -> pd.DataFrame:
    """
    action:
      'drop'      — удалить строки с NaN в col
      'cast'      — привести к числу
      'mean'      — заполнить средним
      'ffill'     — заполнить предыдущим
      'bfill'     — заполнить следующим
      'zero'      — заполнить нулём
      'value'     — заполнить value
    """
    df = df.copy()
    if action == "drop":
        df = df.dropna(subset=[col])
    elif action == "cast":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif action == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif action == "ffill":
        df[col] = df[col].ffill()
    elif action == "bfill":
        df[col] = df[col].bfill()
    elif action == "zero":
        df[col] = df[col].fillna(0)
    elif action == "value":
        df[col] = df[col].fillna(value)
    else:
        raise ValueError(f"Неизвестное действие: {action}")
    return df
def interactive_menu(df: pd.DataFrame) -> pd.DataFrame:
    while True:
        print("\n=== Меню обработки ===")
        print("1. Показать проблемы")
        print("2. Обработать столбец")
        print("3. Выход")
        choice = input("Выбор: ").strip()
        if choice == "1":
            print(check_data_quality(df))
        elif choice == "2":
            col = input("Столбец: ").strip()
            action = input("Действие (drop/cast/mean/ffill/bfill/zero/value): ").strip()
            val = input("Значение (для value): ").strip() if action == "value" else None
            df = handle_missing(df, col, action, val)
            print("Готово.")
        elif choice == "3":
            break
    return df
def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=np.float64)
class MinMaxScalerCustom:
    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a, self.b = a, b
        self.min_, self.max_ = None, None

    def fit(self, x: np.ndarray):
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        # защита от деления на ноль
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return self.a + (x - self.min_) * (self.b - self.a) / denom

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return (x_new - self.a) * denom / (self.b - self.a) + self.min_


class StandardScalerCustom:
    def __init__(self):
        self.mean_, self.std_ = None, None

    def fit(self, x: np.ndarray):
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)  # выборочное СКО
        return self

    def transform(self, x):
        std = np.where(self.std_ == 0, 1, self.std_)
        return (x - self.mean_) / std

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new):
        std = np.where(self.std_ == 0, 1, self.std_)
        return x_new * std + self.mean_
def split_train_val_test(x: np.ndarray, ratios, percent: bool = False):
    """
    ratios: кортеж (r1, r2, r3). Если percent=True — в процентах.
    Возвращает (x_train, x_val, x_test).
    """
    r = np.array(ratios, dtype=float)
    if percent:
        r = r / 100.0
    r = r / r.sum()  # нормируем

    n = len(x)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))
    n_test = n - n_train - n_val  # остаток — в тест

    # равномерное разбиение (без перемешивания, чтобы сохранить порядок)
    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    return x_train, x_val, x_test
