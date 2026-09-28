"""Задания 38-45. Тензорные вычисления в TensorFlow, PyTorch и Keras.

TensorFlow + PyTorch + Keras занимают ~5 ГБ RAM, поэтому фреймворки
загружаются ТОЛЬКО внутри run_tensors() — до тех пор, пока пользователь
не запросил именно этот раздел, они не попадают в память.
"""

import numpy as np

from src.config import run_step, step_done, step_skipped


def _load_frameworks():
    """Ленивая загрузка тяжёлых фреймворков (вызывается один раз)."""
    import tensorflow as tf
    import torch
    import keras
    from keras import ops
    torch.set_num_threads(4)
    tf.get_logger().setLevel("ERROR")
    return tf, torch, keras, ops


def release_frameworks() -> list:
    """Выгружает tensorflow/torch/keras из памяти после заданий 38-45.

    Вместе они занимают несколько ГБ RAM (и часть VRAM, если TensorFlow
    увидел GPU). Задания 46-50 идут после тензорного раздела и тяжёлых
    библиотек не требуют, поэтому после него фреймворки выбрасываются из
    sys.modules вместе с подмодулями, а gc.collect() освобождает память.
    Возвращает список выгруженных модулей.
    """
    import gc
    import sys

    roots = ("tensorflow", "torch", "keras")
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        # освобождаем видеопамять до выгрузки модулей
        try:
            for dev in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(dev, True)
        except Exception:
            pass

    names = [n for n in sys.modules if n.split(".")[0] in roots]
    for name in names:
        sys.modules.pop(name, None)
    gc.collect()
    return names


def run_tensors(data: np.ndarray) -> dict:
    """Задания 38-45 на матрице X формы (n, m)."""
    try:
        tf, torch, keras, ops = _load_frameworks()
    except Exception as exc:  # фреймворки не установлены / не грузятся
        step_skipped(f"tensorflow/torch/keras недоступны: {exc}")
        return {}

    n, m = data.shape
    results = {}

    # ---------------- 38/39. Тензоры из данных ----------------
    run_step(38, "Тензор TensorFlow")
    X_tf = tf.constant(data, dtype=tf.float32)
    results["X_tf"] = (tuple(X_tf.shape), X_tf.dtype.name)
    step_done(f"{X_tf.shape}")

    run_step(39, "Тензор PyTorch")
    X_pt = torch.tensor(data, dtype=torch.float32)
    results["X_pt"] = (tuple(X_pt.shape), str(X_pt.dtype))
    step_done(f"{tuple(X_pt.shape)}")

    # ---------------- 40. Случайные тензоры A, W, B ----------------
    run_step(40, "Случайные тензоры A(5×n), W(m×4), B(5×4)")
    k, p = 5, 4
    A = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
    W = tf.random.normal((m, p))
    B = tf.random.uniform((k, p))
    results["shapes"] = {"A": (k, n), "W": (m, p), "B": (k, p)}
    step_done(f"k={k}, p={p}")

    # ---------------- 41. AXW + B (TensorFlow) ----------------
    run_step(41, "Операция AXW+B (TensorFlow)")
    AX = tf.matmul(tf.cast(A, tf.float32), X_tf)
    AXW = tf.matmul(AX, W)
    result_tf = AXW + B
    results["tf"] = float(tf.reduce_mean(tf.square(result_tf)).numpy())
    step_done(f"форма {tuple(result_tf.shape)}, "
              f"mean‖result‖² = {results['tf']:.4f}")

    # ---------------- 42. AXW + B (PyTorch) ----------------
    run_step(42, "Операция AXW+B (PyTorch)")
    A_pt = torch.randint(0, 10, (k, n), dtype=torch.float32)
    W_pt = torch.randn(m, p)
    B_pt = torch.rand(k, p)
    result_pt = A_pt @ X_pt @ W_pt + B_pt
    results["pt"] = float(result_pt.square().mean().item())
    step_done(f"форма {tuple(result_pt.shape)}, "
              f"mean‖result‖² = {results['pt']:.4f}")

    # Методичка просит в каждом фреймворке свои случайные A, W, B, поэтому
    # числа выше не обязаны совпадать. Дополнительно считаем ту же операцию
    # на ОДНИХ И ТЕХ ЖЕ входах, что и в TensorFlow, — это и есть проверка,
    # что обе библиотеки реализуют AXW + B одинаково.
    same_pt = (torch.tensor(A.numpy(), dtype=torch.float32)
               @ X_pt
               @ torch.tensor(W.numpy(), dtype=torch.float32)
               + torch.tensor(B.numpy(), dtype=torch.float32))
    tf_res = result_tf.numpy()
    pt_res = same_pt.numpy()
    rel = float(np.max(np.abs(pt_res - tf_res)) / (np.max(np.abs(tf_res)) + 1e-12))
    results["tf_vs_pt_rel_diff"] = rel
    print(f"      контроль на общих входах: max|Δ| / max|result| = {rel:.2e} "
          f"-> {'совпадает' if rel < 1e-5 else 'РАСХОЖДЕНИЕ'}")
    assert rel < 1e-5, "TensorFlow и PyTorch разошлись на одних и тех же A, W, B"

    # ---------------- 43. Матрицы T, P, Q ----------------
    run_step(43, "Матрицы T, P, Q (3×10)")
    T = np.random.rand(3, 10)   # (0, 1) -> sqrt безопасен
    P = np.random.rand(3, 10)
    Q = np.random.rand(3, 10)
    results["TPQ"] = T.shape
    step_done(f"{T.shape}")

    # ---------------- 44. Операция в Keras ----------------
    run_step(44, "Операция abs(sin T - exp P·√Q) в Keras")
    T_k = ops.convert_to_tensor(T)
    P_k = ops.convert_to_tensor(P)
    Q_k = ops.convert_to_tensor(Q)
    V_k = ops.abs(ops.sin(T_k) - ops.exp(P_k) * ops.sqrt(Q_k))
    results["V_k"] = float(np.mean(np.asarray(V_k)))
    step_done(f"форма {tuple(V_k.shape)}, mean|V| = {results['V_k']:.4f}")

    # ---------------- 45. Та же операция в PyTorch ----------------
    run_step(45, "Операция в PyTorch")
    T_t = torch.tensor(T)
    P_t = torch.tensor(P)
    Q_t = torch.tensor(Q)
    V_t = torch.abs(torch.sin(T_t) - torch.exp(P_t) * torch.sqrt(Q_t))
    results["V_t"] = float(V_t.mean().item())
    step_done(f"форма {tuple(V_t.shape)}, mean|V| = {results['V_t']:.4f}, "
              f"Keras≈PyTorch: {np.allclose(results['V_k'], results['V_t'], rtol=1e-4)}")

    return results