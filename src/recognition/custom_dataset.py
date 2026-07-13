import os
import glob
import numpy as np

from .sudoku_preprocess import sudoku_to_mnist

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def build_custom_digit_dataset(root_dir, max_images=None, verbose=True):
    """Returns (X, y, meta) where:
        X    - (N, 28, 28, 1) uint8 array, MNIST-format cell crops
        y    - (N,) int array, true digit labels 1-9
        meta - list of dicts with {image_path, row, col} per sample, for
               debugging/error analysis
    """
    images = []
    for i, img_path in enumerate(glob.glob(os.path.join(root_dir, "*.jpg"))):
        if os.path.exists(img_path):
            images.append(img_path)
        if i==max_images:
            break

    X, y, meta = [], [], []
    n_failed = 0

    for img_path in images:
        try:
            img_X, img_y, img_meta = sudoku_to_mnist(img_path)
            X.extend(img_X)
            y.extend(img_y)
            meta.extend(img_meta)

        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [skip] {os.path.basename(img_path)}: {e}")
            continue

    if verbose:
        print(f"Processed {len(images)} images ({n_failed} failed grid detection).")
        print(f"Collected {len(X)} non-empty digit cells.")

    if len(X) == 0:
        return (np.empty((0, 28, 28, 1), dtype=np.uint8),
                np.empty((0,), dtype=np.int64), meta)

    X = np.array(X, dtype=np.uint8)[..., np.newaxis]
    y = np.array(y, dtype=np.int64)

    return X, y, meta