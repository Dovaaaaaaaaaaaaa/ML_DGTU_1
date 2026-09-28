"""Конфигурация проекта: seed, пути, примитивы вывода.

ВАЖНО (память): здесь НЕ импортируются tensorflow/torch. Эти фреймворки
занимают ~5 ГБ RAM и подгружаются только внутри src/tensors.py в момент
запуска заданий 38-45. set_seeds() берёт их из sys.modules: если они уже
загружены в текущем процессе — сид фиксируется честно, если нет — они и не
загружаются зря.
"""

import os
import sys
import random
from datetime import datetime

import numpy as np

# Выставляется ДО первого импорта matplotlib/tensorflow, иначе появятся
# GUI-окна matplotlib и долгие логи TensorFlow.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Ограничиваем потоки: numpy/sklearn/tf/torch по умолчанию распараллеливают
# на все ядра и съедают память на ноутбуке. setdefault — пользователь может
# переопределить значения вручную.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
# TensorFlow не резервирует сразу всю видеопамять (RTX 4060 задействована
# и системой, и приложениями).
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

SEED = 42

DATA_DIR = "data"
OUTPUT_DIR = "output"
DATASET_PATH = os.path.join(DATA_DIR, "dataset_var7.csv")
IMAGE_PATH = os.path.join(DATA_DIR, "image.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Фиксирует seed для random/numpy и (если загружены) tf/torch.

    Тяжёлые фреймворки берутся из sys.modules, чтобы не импортировать их
    при каждом запуске лабораторной.
    """
    random.seed(seed)
    np.random.seed(seed)

    tf = sys.modules.get("tensorflow")
    if tf is not None:
        tf.random.set_seed(seed)

    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def timestamped_filename(base: str, ext: str) -> str:
    """output/{base}_{ГГГГ-ММ-ДД_ЧЧ-ММ-СС}.{ext} — каждый прогон создаёт
    новый файл и не перетирает прошлые результаты."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_DIR, f"{base}_{ts}.{ext}")


# ----------------------------------------------------------------------
# Примитивы вывода. Каждое задание печатается одной строкой:
#   [Задание  1] Чтение файлов ... OK (dataset_var7.csv -> 500x7)
# ----------------------------------------------------------------------
def banner(text: str) -> None:
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70, flush=True)


def run_step(n: int, title: str) -> None:
    print(f"[Задание {n:>2}] {title} ... ", end="", flush=True)


def step_done(info: str = "") -> None:
    print("OK" + (f" ({info})" if info else ""), flush=True)


def step_skipped(reason: str = "") -> None:
    print("ПРОПУЩЕНО" + (f" ({reason})" if reason else ""), flush=True)