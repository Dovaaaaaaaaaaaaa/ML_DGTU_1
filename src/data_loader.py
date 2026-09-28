"""Задания 4-7. Качество данных, вывод таблицы, извлечение столбцов, типы."""

import pandas as pd


def check_data_quality(df: pd.DataFrame) -> dict:
    """Отчёт о пропусках, типах и «испорченных» нечисловых ячейках.

    Ключи отчёта: shape, missing, dtypes, non_numeric.
    non_numeric[col] = {'examples': [...], 'numeric_ratio': float}.
    numeric_ratio — доля реально числовых значений столбца: у x1 ~0.996
    (числовой с битыми ячейками), у category ~0.0 (честный текст).
    """
    report = {
        "shape": df.shape,
        "missing": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "non_numeric": {},
    }

    for col in df.columns:
        # числовые и даты пропускаем
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # пробуем привести к числу: «превратилось в NaN, но изначально не было
        # пустым» = именно испорченная ячейка (например, текст 'ошибка')
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = df[col][converted.isna() & df[col].notna()]
        if not bad.empty:
            ratio = 1.0 - len(bad) / int(df[col].notna().sum())
            report["non_numeric"][col] = {
                "examples": bad.tolist()[:10],
                "numeric_ratio": round(ratio, 4),
            }
    return report


def print_quality_report(report: dict) -> None:
    """Форматированный вывод отчёта из check_data_quality."""
    print(f"      форма: {report['shape']}")
    print(f"      типы : {report['dtypes']}")

    miss = {k: v for k, v in report["missing"].items() if v}
    if miss:
        print(f"      пропуски: {miss}")
    else:
        print("      пропуски: нет")

    for col, info in report["non_numeric"].items():
        kind = ("числовой с испорченными ячейками"
                if info["numeric_ratio"] > 0 else "текстовый")
        print(f"      {col}: {kind}, числовой доли {info['numeric_ratio']}, "
              f"примеры: {info['examples']}")


def show_table(df: pd.DataFrame, n: int = 10) -> None:
    """Вывод первых n строк, размерности и типов."""
    print(df.head(n))
    print(f"\nShape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")


def extract_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Возвращает копию DataFrame с нужными столбцами."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    return df[cols].copy()  # копия — исходный df не меняется


def cast_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """Приведение типов. type_map: {'col': 'float64', ...}.

    Для числовых типов используется pd.to_numeric(errors='coerce'), чтобы
    текст ('ошибка' в x1) превратился в NaN, а не уронил скрипт ValueError,
    как это произошло бы с наивным .astype() в pandas 3 (StringDtype).
    """
    df = df.copy()
    for col, dtype in type_map.items():
        if col not in df.columns:
            raise KeyError(f"Нет столбца: {col}")
        if dtype.startswith(("int", "float")):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        else:
            df[col] = df[col].astype(dtype)
    return df