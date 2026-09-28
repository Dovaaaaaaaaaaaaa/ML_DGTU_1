"""Задание 50. Обработка изображения средствами SciPy.

10 подзаданий методички:
  1. Разделение цветовых каналов R/G/B
  2. Поворот на 40 градусов
  3. Преобразование в чёрно-белое
  4. Гистограмма ч/б изображения (256 бинов)
  5. Разбиение на 3 части по значениям гистограммы
  6. Чёрная круговая рамка
  7. Гауссовский шум (sigma = 25)
  8. Гауссовское размытие (sigma = 2)
  9. Sharpening (unsharp masking, amount = 1.5)
  10. PCA к изображению (восстановление, кривая MSE, сетка компонент)
"""

# Десять подзаданий задания 50 — используется для отчёта о выполнении.
SUBTASKS = (
    "разделение цветовых каналов R/G/B",
    "поворот на 40 градусов",
    "перевод в чёрно-белое",
    "гистограмма ч/б изображения (256 бинов)",
    "разбиение на 3 части по гистограмме",
    "чёрная круговая рамка",
    "гауссовский шум (sigma = 25)",
    "гауссовское размытие (sigma = 2)",
    "sharpening (amount = 1.5)",
    "PCA по блокам 8x8",
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from sklearn.decomposition import PCA
from skimage.color import rgb2gray

from src.io_utils import save_figure


def _prep_image(image_path: str) -> np.ndarray:
    """Читает изображение, приводит к RGB uint8 и ограничивает до 1024 px.

    PCA по пикселям на миллионах точек съедает память, поэтому по
    рекомендации методички работать с изображениями больше 1024×1024
    не будем.
    """
    img = plt.imread(image_path)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]  # отбрасываем альфа-канал
    if not np.issubdtype(img.dtype, np.uint8):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    if scale < 1.0:
        from skimage.transform import resize
        img = resize(img, (int(h * scale), int(w * scale)),
                     preserve_range=True, anti_aliasing=True).astype(np.uint8)
    print(f"      изображение приведено к RGB uint8, размер {img.shape}")
    return img


def split_channels(img: np.ndarray) -> dict:
    """1. Разделение цветного изображения на каналы R, G, B."""
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается цветное изображение (H, W, 3)")

    r, g, b = img[..., 0].copy(), img[..., 1].copy(), img[..., 2].copy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img);   axes[0].set_title("Исходное")
    axes[1].imshow(r, cmap="Reds");    axes[1].set_title("Красный канал")
    axes[2].imshow(g, cmap="Greens");  axes[2].set_title("Зелёный канал")
    axes[3].imshow(b, cmap="Blues");   axes[3].set_title("Синий канал")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "task50_channels")
    plt.close(fig)
    return {"r": r, "g": g, "b": b}


def rotate_image(img: np.ndarray, angle: float = 40.0) -> np.ndarray:
    """2. Поворот на угол (бикубическая интерполяция, холст не обрезается)."""
    rotated = rotate(img.astype(np.float32), angle=angle, reshape=True,
                     order=3, mode="constant", cval=0)
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);      axes[0].set_title("Исходное")
    axes[1].imshow(rotated);  axes[1].set_title(f"Поворот на {angle}°")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_rotated")
    plt.close(fig)
    return rotated


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """3. Чёрно-белое через взвешенную яркость (Rec. 601/709)."""
    gray = rgb2gray(img)
    gray_u8 = (gray * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);                  axes[0].set_title("Исходное")
    axes[1].imshow(gray_u8, cmap="gray"); axes[1].set_title("Ч/б")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_gray")
    plt.close(fig)
    return gray_u8


def gray_histogram(gray: np.ndarray, bins: int = 256) -> dict:
    """4. Гистограмма яркостей ч/б изображения (256 бинов)."""
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], hist, width=1, color="black", align="edge")
    ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Частота")
    ax.set_xlim(0, 255)
    save_figure(fig, "task50_gray_hist")
    plt.close(fig)
    return {"hist": hist, "bin_edges": bin_edges}


def split_by_histogram(gray: np.ndarray, hist_info: dict) -> tuple:
    """5. Разбиение на 3 части по квантилям гистограммы (тёмные/средние/светлые)."""
    hist = hist_info["hist"]
    edges = hist_info["bin_edges"]

    cum = np.cumsum(hist)
    total = cum[-1]
    t1 = edges[np.searchsorted(cum, total * 0.33)]
    t2 = edges[np.searchsorted(cum, total * 0.66)]

    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(part1, cmap="gray"); axes[0].set_title(f"Тёмные (≤ {t1})")
    axes[1].imshow(part2, cmap="gray"); axes[1].set_title(f"Средние ({t1}–{t2})")
    axes[2].imshow(part3, cmap="gray"); axes[2].set_title(f"Светлые (> {t2})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_split3")
    plt.close(fig)
    return part1, part2, part3


def add_circular_frame(img: np.ndarray) -> np.ndarray:
    """6. Чёрная круговая рамка (всё вне круга зануляется)."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    R = min(h, w) / 2.0 - 2

    Y, X = np.ogrid[:h, :w]
    mask_outside = (X - cx) ** 2 + (Y - cy) ** 2 > R ** 2

    framed = img.copy()
    if framed.ndim == 2:
        framed[mask_outside] = 0
    else:
        framed[mask_outside, :] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);    axes[0].set_title("Исходное")
    axes[1].imshow(framed); axes[1].set_title("В круговой рамке")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_frame")
    plt.close(fig)
    return framed


def add_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """7. Гауссовский шум с нулевым средним и заданным СКО."""
    noise = np.random.normal(0.0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);   axes[0].set_title("Исходное")
    axes[1].imshow(noisy); axes[1].set_title(f"Шум (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_noisy")
    plt.close(fig)
    return noisy


def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """8. Гауссовское размытие (по каждому каналу для цветного)."""
    if img.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
             for c in range(img.shape[2])],
            axis=-1,
        )
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);     axes[0].set_title("Исходное")
    axes[1].imshow(blurred); axes[1].set_title(f"Размытие (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_blurred")
    plt.close(fig)
    return blurred


def sharpen_image(img: np.ndarray, sigma: float = 2.0,
                  amount: float = 1.5) -> np.ndarray:
    """9. Sharpening через unsharp masking: orig + amount·(orig - blur)."""
    blurred = gaussian_filter(img.astype(np.float32),
                              sigma=(sigma, sigma, 0) if img.ndim == 3 else sigma)
    sharpened = np.clip(
        img.astype(np.float32) + amount * (img.astype(np.float32) - blurred),
        0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);       axes[0].set_title("Исходное")
    axes[1].imshow(sharpened); axes[1].set_title(f"Sharpening (amount={amount})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_sharpened")
    plt.close(fig)
    return sharpened


def _to_patches(img: np.ndarray, block: int = 8) -> np.ndarray:
    """Нарезает изображение на неперекрывающиеся блоки block x block.

    Методичка рекомендует именно такой вариант: «для сжатия пространственных
    патчей разбейте изображение на блоки 8x8 и подайте их как вектор из
    64 признаков». Каждый блок RGB даёт вектор из 8*8*3 = 192 признаков,
    блок ч/б — 8*8 = 64.
    """
    h, w = img.shape[:2]
    hh, ww = h // block * block, w // block * block
    trimmed = img[:hh, :ww]
    chans = trimmed.shape[2] if trimmed.ndim == 3 else 1
    if trimmed.ndim == 2:
        trimmed = trimmed[..., None]
    # (h/b, b, w/b, b, c) -> (h/b, w/b, b*b*c)
    grid = trimmed.reshape(hh // block, block, ww // block, block, chans)
    patches = grid.transpose(0, 2, 1, 3, 4).reshape(-1, block * block * chans)
    return patches.astype(np.float32)


def _from_patches(patches: np.ndarray, img_shape: tuple, block: int = 8) -> np.ndarray:
    """Обратная сборка изображения из матрицы блоков."""
    h, w = img_shape[:2]
    chans = img_shape[2] if len(img_shape) == 3 else 1
    hh, ww = h // block * block, w // block * block
    nb_h, nb_w = hh // block, ww // block
    grid = patches.reshape(nb_h, nb_w, block, block, chans)
    img = grid.transpose(0, 2, 1, 3, 4).reshape(hh, ww, chans)
    if len(img_shape) == 2:
        img = img[..., 0]
    return img


def apply_pca_to_image(img: np.ndarray, n_components: int = 32,
                       block: int = 8) -> dict:
    """10. PCA к изображению.

    Считаем PCA по блокам block x block (рекомендация методички), потому что
    при PCA по пикселям у цветного изображения всего 3 признака, и число
    компонент не может превысить 3 — такое PCA ничего не сжимает. По блокам
    признаков block*block*channels, поэтому n_components = 32 осмысленно.
    """
    patches = _to_patches(img, block=block)
    n_features = patches.shape[1]
    results = {}

    # ---- PCA по пикселям (вариант из основного текста методички) ----
    h, w = img.shape[:2]
    if img.ndim == 3:
        flat_px = img.reshape(-1, 3).astype(np.float32)
    else:
        flat_px = img.reshape(-1, 1).astype(np.float32)
    k_px = min(3, flat_px.shape[1])
    pca_px = PCA(n_components=k_px).fit(flat_px)
    results["pixel_pca"] = {
        "n_features": int(flat_px.shape[1]),
        "n_components": k_px,
        "explained_variance_ratio": pca_px.explained_variance_ratio_,
    }
    print(f"      PCA по пикселям: {flat_px.shape[1]} признак(ов), "
          f"{k_px} компонент(ы) — сжатия почти нет")

    # ---- PCA по блокам (основной, осмысленный вариант) ----
    k = int(min(n_components, n_features, patches.shape[0]))
    pca = PCA(n_components=k)
    restored_flat = pca.inverse_transform(pca.fit_transform(patches))
    restored = _from_patches(restored_flat, img.shape, block=block)
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
    axes[0].set_title("Исходное")
    axes[1].imshow(restored, cmap="gray" if img.ndim == 2 else None)
    axes[1].set_title(f"PCA по блокам {block}x{block} ({k} из {n_features})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "task50_pca_restored")
    plt.close(fig)

    # ---- Кривая MSE от числа компонент ----
    # Ограничиваем 32: при изображении 1024x1024 блоков уже 16384, и полный
    # перебор до n_features съел бы минуты ради графика, который в отчёте
    # всё равно читается только до ~32.
    max_k = int(min(n_features, patches.shape[0], 32))
    mse_values = []
    for kk in range(1, max_k + 1):
        pca_k = PCA(n_components=kk)
        r = pca_k.inverse_transform(pca_k.fit_transform(patches))
        mse_values.append(float(np.mean((patches - r) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_k + 1), mse_values, marker="o")
    ax.set_xlabel("Число главных компонент")
    ax.set_ylabel("MSE восстановления блоков")
    ax.set_title(f"MSE от числа компонент ({max_k} признаков)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "task50_pca_mse")
    plt.close(fig)

    # ---- Первые 16 главных компонент как изображения block x block ----
    n_show = int(min(16, pca.components_.shape[0]))
    chans = 3 if img.ndim == 3 else 1
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        if i < n_show:
            comp = pca.components_[i].reshape(block, block, chans)
            ax.imshow(comp[..., 0] if chans == 1 else comp, cmap="gray")
            ax.set_title(f"PC {i + 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Первые {n_show} главных компонент")
    fig.tight_layout()
    save_figure(fig, "task50_pca_components")
    plt.close(fig)

    results.update({
        "restored": restored,
        "n_features": int(n_features),
        "n_components": k,
        "block": block,
        "mse_curve": mse_values,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    print(f"      PCA по блокам {block}x{block}: {n_features} признаков, "
          f"{k} компонент, MSE(1) = {mse_values[0]:.2f} -> "
          f"MSE({k}) = {mse_values[k - 1]:.4f}")
    return results


def make_demo_image(path: str, size: int = 512) -> None:
    """Генерация демонстрационного изображения, если файла нет."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = (np.sin(xx / 40.0) * 127 + 128).astype(np.uint8)
    g = (np.sin(yy / 50.0) * 127 + 128).astype(np.uint8)
    b = (((xx + yy) * 0.5) % 256).astype(np.uint8)
    rgb = np.dstack([r, g, b])
    plt.imsave(path, rgb)
    print(f"      создано демонстрационное изображение: {path}")


def run_task50(image_path: str) -> dict:
    """Последовательно выполняет все 10 подзаданий."""
    img = _prep_image(image_path)

    results = {}
    results["channels"] = split_channels(img)
    results["rotated"] = rotate_image(img, angle=40.0)
    results["gray"] = to_grayscale(img)
    results["hist"] = gray_histogram(results["gray"])
    results["parts"] = split_by_histogram(results["gray"], results["hist"])
    results["framed"] = add_circular_frame(img)
    results["noisy"] = add_noise(img, sigma=25.0)
    results["blurred"] = gaussian_blur(img, sigma=2.0)
    results["sharpened"] = sharpen_image(img, sigma=2.0, amount=1.5)
    results["pca"] = apply_pca_to_image(img, n_components=32, block=8)

    print("      Задание 50 выполнено. Все графики сохранены в каталог output/.")
    return results