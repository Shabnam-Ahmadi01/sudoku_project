import os
import glob
import numpy as np

from .sudoku_preprocess import sudoku_to_mnist

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def build_custom_digit_dataset(root_dir, max_images=None, empty_fraction=0.1, verbose=True):
    """Returns (X, y, meta) where:
        X    - (N, 28, 28, 1) uint8 array, MNIST-format cell crops
        y    - (N,) int array, labels: 0=empty cell, 1-9=sudoku digits
        meta - list of dicts with {image_path, row, col} per sample

    empty_fraction: ratio of empty (class-0) cells to include relative to
        the number of non-empty cells, e.g. 0.1 keeps ~10% extra blank cells.
        Set to 0.0 to exclude empty cells entirely.
    """
    images = []
    for i, img_path in enumerate(glob.glob(os.path.join(root_dir, "*.jpg"))):
        if os.path.exists(img_path):
            images.append(img_path)
        if max_images is not None and i + 1 >= max_images:
            break

    X_digits, y_digits, meta_digits = [], [], []
    X_empty,  y_empty,  meta_empty  = [], [], []
    n_failed = 0

    for img_path in images:
        try:
            img_X, img_y, img_meta = sudoku_to_mnist(img_path, include_empty=(empty_fraction > 0))
            for cell, label, m in zip(img_X, img_y, img_meta):
                if label == 0:
                    X_empty.append(cell)
                    y_empty.append(0)
                    meta_empty.append(m)
                else:
                    X_digits.append(cell)
                    y_digits.append(label)
                    meta_digits.append(m)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [skip] {os.path.basename(img_path)}: {e}")
            continue

    # Sample a controlled proportion of empty cells
    n_keep_empty = int(len(X_digits) * empty_fraction)
    if n_keep_empty > 0 and len(X_empty) > 0:
        idx = np.random.choice(len(X_empty), min(n_keep_empty, len(X_empty)), replace=False)
        X_sel    = [X_empty[i]    for i in idx]
        y_sel    = [y_empty[i]    for i in idx]
        meta_sel = [meta_empty[i] for i in idx]
    else:
        X_sel, y_sel, meta_sel = [], [], []

    X    = X_digits    + X_sel
    y    = y_digits    + y_sel
    meta = meta_digits + meta_sel

    if verbose:
        n_dig = len(X_digits)
        n_emp = len(X_sel)
        print(f"Processed {len(images)} images ({n_failed} failed grid detection).")
        print(f"Collected {n_dig} digit cells and {n_emp} empty cells (class 0).")

    if len(X) == 0:
        return (np.empty((0, 28, 28, 1), dtype=np.uint8),
                np.empty((0,), dtype=np.int64), meta)

    X = np.array(X, dtype=np.uint8)[..., np.newaxis]
    y = np.array(y, dtype=np.int64)

    return X, y, meta
