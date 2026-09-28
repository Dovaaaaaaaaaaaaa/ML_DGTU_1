"""Точка входа лабораторной работы №1.

«Библиотеки Python для машинного обучения. Работа с данными»

Запускает все 50 заданий в порядке методички:
    Раздел 1 (1-7)   чтение/запись данных
    Раздел 2 (8-13)  предобработка
    Раздел 3 (14-33) статистический анализ
    34-35            масштабирование
    36-37            скользящее окно
    38-45            тензорные вычисления
    46-49            матричные операции и пайплайн
    50               обработка изображения

Запуск:
    python main.py                     # все 50 заданий
    python main.py --sections stats    # только выбранные разделы
    python main.py --interactive       # + интерактивное меню задания 8
    python main.py --list-sections     # что во что входит

Разделы io (1-7) и prep (8-13) выполняются всегда: из них берётся
датасет для остальных разделов. Остальные включаются через --sections.

Память: тяжёлые фреймворки (TensorFlow, PyTorch, Keras) грузятся только
в разделе tensors и выгружаются сразу после него, поэтому задания 46-50
не стартуют на пиковом расходе памяти.
"""

import argparse
import importlib
import os
import sys
import time

# src.config первым: он выставляет MPLBACKEND=Agg и TF_CPP_MIN_LOG_LEVEL=3
# до импорта matplotlib/tensorflow, иначе открываются окна GUI.
from src.config import (
    SEED,
    DATASET_PATH,
    IMAGE_PATH,
    OUTPUT_DIR,
    banner,
    run_step,
    step_done,
    set_seeds,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.io_utils import read_table, write_table, save_figure
from src.data_loader import (
    check_data_quality,
    print_quality_report,
    show_table,
    extract_columns,
    cast_types,
)
from src.preprocess import (
    handle_missing,
    interactive_menu,
    to_numpy,
    split_train_val_test,
    MinMaxScalerCustom,
    StandardScalerCustom,
)

# Модули разделов 3-6 не импортируются заранее: stats/scaling/windows/matrix/image
# тянут за собой sklearn, SciPy и scikit-image, а tensors — ещё и TensorFlow с
# PyTorch. Импорт по требованию позволяет запускать лёгкие разделы, не загружая
# в память то, что в них не участвует.
#
# Ключ --sections -> (модуль src, описание). io и prep сюда не входят: они
# выполняются всегда, потому что из них берутся данные для остальных.
SECTIONS = (
    ("stats", "stats_analysis", "задания 14-33, статистический анализ"),
    ("scaling", "scaling", "задания 34-35, масштабирование vs sklearn"),
    ("windows", "windows", "задания 36-37, скользящее окно и среднее"),
    ("tensors", "tensors", "задания 38-45, TensorFlow/PyTorch/Keras (тяжёлый)"),
    ("matrix", "matrix_ops", "задания 46-49, разрежённые матрицы и PCA"),
    ("image", "image_proc", "задание 50, обработка изображения"),
)
SECTION_MODULES = {key: module for key, module, _ in SECTIONS}


def get_module(name: str):
    """Импортирует модуль раздела при первом обращении к нему."""
    if name not in SECTION_MODULES:
        raise ValueError(f"Неизвестный раздел: {name}")
    return importlib.import_module(f"src.{SECTION_MODULES[name]}")


# Столбцы варианта 7
FEATURE_COLS = ["x1", "x2", "x3", "y"]


# ======================================================================
# Раздел 1. Считывание и запись данных (задания 1-7)
# ======================================================================
def section_io(df_path: str, state: dict) -> None:
    banner("Раздел 1. Считывание и запись данных")

    # ---------------- Задание 1. Чтение файлов ----------------
    run_step(1, "Чтение файлов")
    df = read_table(df_path)
    state["df_raw"] = df
    step_done(f"{os.path.basename(df_path)} -> {df.shape[0]}x{df.shape[1]}")

    # ---------------- Задание 2. Запись файлов ----------------
    run_step(2, "Запись файлов")
    # Во все форматы пишем числовую таблицу FEATURE_COLS: savemat не умеет
    # сохранять массивы со строковыми значениями (timestamp, category).
    df_num = _clean_features(df)
    paths = write_table(df_num, "task02_dataset")
    state["write_paths"] = paths
    step_done(", ".join(f"{k}" for k in paths))

    # проверка кругового преобразования: читаем обратно и сверяем
    for ext, p in paths.items():
        back = read_table(p)
        same_shape = back.shape == df_num.shape
        diff = float("nan")
        ok = same_shape and all(str(c) in back.columns for c in df_num.columns)
        if ok:
            diff = float(np.nanmax(np.abs(back[list(df_num.columns)].to_numpy(dtype=float)
                                          - df_num.to_numpy(dtype=float))))
            ok = diff < 1e-9
        print(f"      {ext:<5} {os.path.basename(p)}: "
              f"форма {back.shape} {'совпала' if same_shape else 'НЕ совпала'}, "
              f"max|Δ| = {diff:.2e} -> {'OK' if ok else 'РАСХОЖДЕНИЕ'}")
        if not ok:
            raise AssertionError(f"круговое преобразование не сошлось для .{ext}")

    # ---------------- Задание 3. Сохранение изображений ----------------
    run_step(3, "Сохранение изображений")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(df["x2"].to_numpy(dtype=float), label="x2 (с пропусками)", linewidth=0.8)
    axes[0].plot(df["x3"].to_numpy(dtype=float), label="x3", linewidth=0.8)
    axes[0].set_title("Исходные данные: x2 и x3")
    axes[0].set_xlabel("Индекс")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(df["y"].to_numpy(dtype=float), bins=40, color="#4C72B0",
                 edgecolor="black", alpha=0.8)
    axes[1].set_title("Распределение y")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Частота")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p_fig = save_figure(fig, "task03_saved_figure")
    plt.close(fig)
    step_done(os.path.basename(p_fig))

    # ---------------- Задание 4. Проверка данных ----------------
    run_step(4, "Проверка качества данных")
    report = check_data_quality(df)
    state["report"] = report
    print_quality_report(report)
    step_done(f"пропусков: {sum(report['missing'].values())}, "
              f"столбцов с мусором: {len(report['non_numeric'])}")

    # ---------------- Задание 5. Вывод таблицы ----------------
    run_step(5, "Вывод таблицы")
    print()
    show_table(df, n=5)
    print()
    step_done()

    # ---------------- Задание 6. Извлечение столбцов по варианту ----------------
    run_step(6, f"Извлечение столбцов варианта {FEATURE_COLS}")
    df_sel = extract_columns(df, FEATURE_COLS)
    # исходный DataFrame не должен измениться
    assert df.shape[1] == report["shape"][1], "extract_columns изменил исходный df"
    assert df_sel is not df, "extract_columns вернул не копию"
    step_done(f"{df_sel.shape[0]}x{df_sel.shape[1]} (исходный df не изменён)")

    # ---------------- Задание 7. Явные типы данных ----------------
    run_step(7, "Явные типы данных")
    # x1 сначала приводим к float64: в исходном файле он строковый,
    # потому что в двух ячейках лежит текст "ошибка" (см. задание 4).
    df_typed = cast_types(df_sel, {"x1": "float64", "x2": "float64",
                                   "x3": "float64", "y": "float64"})
    state["df_typed"] = df_typed
    print("      было :", dict(df_sel.dtypes.astype(str)))
    print("      стало:", dict(df_typed.dtypes.astype(str)))
    print("      category осталась строкой и не приводится к числу;")
    print("      timestamp переводится в datetime64[ns]:")
    df_dates = cast_types(df, {"timestamp": "datetime64[ns]"})
    print("      ", df_dates["timestamp"].dtype, "| min =", df_dates["timestamp"].min())
    print("      int64 / category / datetime64[ns] тоже доступны:")
    df_extra = cast_types(df, {"id": "int64", "category": "category"})
    print("      ", dict(df_extra[["id", "category"]].dtypes.astype(str)))
    step_done()


# ======================================================================
# Раздел 2. Предобработка данных (задания 8-13)
# ======================================================================
def section_preprocess(df: pd.DataFrame, state: dict, interactive: bool) -> None:
    banner("Раздел 2. Предобработка данных")

    df_sel = extract_columns(df, FEATURE_COLS)
    df_num = cast_types(df_sel, {"x1": "float64", "x2": "float64",
                                 "x3": "float64", "y": "float64"})

    # ---------------- Задание 8. Меню обработки проблем ----------------
    run_step(8, "Меню обработки проблемных значений")
    n_junk = int(df_num["x1"].isna().sum())
    n_nan = int(df_num["x2"].isna().sum())
    print(f"      найдено: x1 - {n_junk} ячеек не приводятся к числу (текст 'ошибка'), "
          f"x2 - {n_nan} пропусков")
    print("      применяем: x1 -> 'cast' (привести к числу), затем 'mean' (заполнить средним)")
    df_fixed = handle_missing(df_num, "x1", "cast")
    df_fixed = handle_missing(df_fixed, "x1", "mean")
    df_fixed = handle_missing(df_fixed, "x2", "mean")
    left = int(df_fixed.isna().sum().sum())
    assert left == 0, f"осталось {left} пропусков"
    print(f"      после обработки пропусков не осталось; "
          f"x1 mean = {df_fixed['x1'].mean():.6f}, x2 mean = {df_fixed['x2'].mean():.6f}")

    # все остальные действия меню показываем на примере, чтобы было видно,
    # что реализованы все семь вариантов из задания
    demo_col = "x2"
    for action in ("drop", "ffill", "bfill", "zero", "value"):
        if action == "value":
            out = handle_missing(df_num, demo_col, action, 0.0)
        elif action == "drop":
            out = handle_missing(df_num, demo_col, action)
        else:
            out = handle_missing(df_num, demo_col, action)
        print(f"      действие '{action:<5}' -> осталось пропусков: "
              f"{int(out[demo_col].isna().sum())}, строк: {len(out)}")

    if interactive:
        print("      включаем интерактивное меню (--interactive)")
        df_fixed = interactive_menu(df_fixed)
    state["df_fixed"] = df_fixed
    step_done()

    # ---------------- Задание 9. DataFrame -> NumPy ----------------
    run_step(9, "DataFrame -> NumPy")
    data = to_numpy(df_fixed)
    state["data"] = data
    assert data.dtype == np.float64 and not np.isnan(data).any()
    step_done(f"{data.shape}, dtype {data.dtype}, NaN нет")

    # ---------------- Задание 10. Функции масштабирования ----------------
    run_step(10, "Собственные функции масштабирования")
    mm = MinMaxScalerCustom(a=0.0, b=1.0).fit(data)
    x_mm = mm.transform(data)
    std = StandardScalerCustom().fit(data)
    x_std = std.transform(data)
    print(f"      MinMax: каждый столбец в [{x_mm.min():.4f}, {x_mm.max():.4f}]")
    print(f"      Standard: средние = {np.round(x_std.mean(axis=0), 10)}, "
          f"СКО = {np.round(x_std.std(axis=0), 4)}")
    assert np.allclose(x_mm.min(axis=0), 0.0) and np.allclose(x_mm.max(axis=0), 1.0)
    assert np.allclose(x_std.mean(axis=0), 0.0, atol=1e-10)
    step_done()

    # ---------------- Задание 11. Восстановление масштаба ----------------
    run_step(11, "Восстановление масштаба (inverse_transform)")
    back_mm = mm.inverse_transform(x_mm)
    back_std = std.inverse_transform(x_std)
    e_mm = float(np.max(np.abs(back_mm - data)))
    e_std = float(np.max(np.abs(back_std - data)))
    assert e_mm < 1e-10 and e_std < 1e-10
    print(f"      MinMax:    max|x - x'| = {e_mm:.3e}")
    print(f"      Standard:  max|x - x'| = {e_std:.3e}")
    state["scalers"] = {"minmax": mm, "std": std, "x_mm": x_mm, "x_std": x_std}
    step_done()

    # ---------------- Задание 12. Разбиение на 3 части ----------------
    run_step(12, "Разбиение на 3 части")
    tr, va, te = split_train_val_test(data, (60, 20, 20), percent=True)
    print(f"      пропорции 60/20/20 -> train {tr.shape}, val {va.shape}, test {te.shape}")
    assert len(tr) + len(va) + len(te) == len(data)
    # вариант со случайным перемешиванием через np.random.permutation
    idx = np.random.permutation(len(data))
    tr2, va2, te2 = split_train_val_test(data[idx], (0.6, 0.2, 0.2))
    print(f"      со случайным перемешиванием -> train {tr2.shape}, "
          f"val {va2.shape}, test {te2.shape}")
    state["split"] = {"train": tr, "val": va, "test": te}
    step_done()

    # ---------------- Задание 13. Фиксация seed ----------------
    run_step(13, "Фиксация seed")
    a = np.random.rand(3)
    set_seeds(SEED)
    b = np.random.rand(3)
    set_seeds(SEED)
    c = np.random.rand(3)
    assert np.array_equal(b, c), "seed не фиксируется"
    print(f"      SEED = {SEED}; два вызова подряд дают одинаковые числа: "
          f"{np.array_equal(b, c)} ({np.round(b, 6)})")
    step_done()


# ======================================================================
# Вспомогательное
# ======================================================================
def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает числовую таблицу FEATURE_COLS без пропусков и мусора."""
    d = extract_columns(df, FEATURE_COLS)
    d = cast_types(d, {c: "float64" for c in FEATURE_COLS})
    for col in ("x1", "x2"):
        d = handle_missing(d, col, "cast")
        d = handle_missing(d, col, "mean")
    return d


# ======================================================================
# main
# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Лабораторная работа №1")
    ap.add_argument("--dataset", default=DATASET_PATH, help="путь к датасету")
    ap.add_argument("--image", default=IMAGE_PATH, help="путь к изображению для задания 50")
    ap.add_argument("--interactive", action="store_true",
                    help="включить интерактивное меню задания 8")
    ap.add_argument("--sections", nargs="*", default=None,
                    help="разделы, которые нужно выполнить: stats scaling windows "
                         "tensors matrix image (разделы io и prep выполняются "
                         "всегда — из них берутся данные для остальных)")
    ap.add_argument("--list-sections", action="store_true",
                    help="показать список разделов и выйти")
    args = ap.parse_args()

    if args.list_sections:
        for key, _, desc in SECTIONS:
            print(f"  {key:<9} {desc}")
        print("\nРазделы io (1-7) и prep (8-13) выполняются всегда.")
        return 0

    if args.sections:
        unknown = [s for s in args.sections if s not in SECTION_MODULES]
        if unknown:
            print(f"ОШИБКА: неизвестные разделы: {', '.join(unknown)}")
            print(f"Доступны: {', '.join(SECTION_MODULES)}")
            return 2

    # Исправление: в консоли Windows по умолчанию cp1251, и печать символа Δ
    # (max|Δ| в задании 2) или ° в задании 50 роняла скрипт с UnicodeEncodeError.
    # Принудительно переводим потоки вывода в UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if not os.path.exists(args.dataset):
        print(f"ОШИБКА: не найден датасет {args.dataset}")
        return 1

    t0 = time.time()
    # Задание 13 требует фиксировать seed в самом начале main.py
    set_seeds(SEED)

    print("=" * 70)
    print("Лабораторная работа №1")
    print("«Библиотеки Python для машинного обучения. Работа с данными»")
    print("=" * 70)
    print(f"Датасет : {args.dataset}")
    print(f"Вывод   : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Seed    : {SEED}")

    state: dict = {}
    sections = args.sections

    def want(name: str) -> bool:
        return sections is None or name in sections

    # --- Разделы 1-2 обязательны: из них берутся данные для остальных ---
    section_io(args.dataset, state)
    section_preprocess(state["df_raw"], state, args.interactive)
    data = state["data"]

    if want("stats"):
        banner("Раздел 3. Работа с данными (задания 14-33)")
        state["stats"] = get_module("stats").run_statistics(state["df_fixed"])

    if want("scaling"):
        banner("Задания 34-35. Сравнение собственного и sklearn-масштабирования")
        state["scaling"] = get_module("scaling").run_scaling(state["df_fixed"])

    if want("windows"):
        banner("Задания 36-37. Скользящее окно и скользящее среднее")
        state["windows"] = get_module("windows").run_windows(data, width=24, col_index=1)

    if want("tensors"):
        banner("Задания 38-45. Тензорные вычисления")
        tensors_mod = get_module("tensors")
        state["tensors"] = tensors_mod.run_tensors(data)
        # TensorFlow + PyTorch + Keras держат несколько ГБ RAM и VRAM, а
        # идущие следом задания 46-50 они не используют — выгружаем их,
        # чтобы тяжёлые разделы не стартовали на максимуме памяти.
        freed = tensors_mod.release_frameworks()
        if freed:
            print(f"      tensorflow/torch/keras выгружены "
                  f"({len(freed)} модулей), память освобождена")

    if want("matrix"):
        banner("Задания 46-49. Матричные операции и пайплайн")
        state["matrix"] = get_module("matrix").run_matrix_ops(data)

    if want("image"):
        banner("Задание 50. Обработка изображения средствами SciPy")
        run_step(50, "Обработка изображения (10 подзаданий)")
        image_proc = get_module("image")
        img_path = args.image
        if not os.path.exists(img_path):
            print(f"      изображения {img_path} нет - генерируем демонстрационное")
            image_proc.make_demo_image(img_path)
        state["image"] = image_proc.run_task50(img_path)
        n_sub = len(image_proc.SUBTASKS)
        step_done(f"{n_sub} подзаданий, {len(os.listdir(OUTPUT_DIR))} файлов в output/")

    dt = time.time() - t0
    n_png = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
    banner(f"Готово за {dt:.1f} с")
    print(f"Файлов в {OUTPUT_DIR}/: {n_png} png, "
          f"{len([f for f in os.listdir(OUTPUT_DIR) if not f.endswith('.png')])} прочих")
    return 0


if __name__ == "__main__":
    sys.exit(main())
