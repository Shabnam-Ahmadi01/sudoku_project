import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .dat_parser import parse_dat_file
from src.preprocessing import process_image
from config import MNIST_SIZE, DIGIT_BOX

def cell_to_mnist_format(cell_gray):
    """Returns a (28, 28) uint8 array in MNIST convention, or None if no
    digit pixels are found."""

    if cell_gray is None or cell_gray.size == 0:
        return None

    blur = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    digit = binary[y1:y2 + 1, x1:x2 + 1]

    h, w = digit.shape
    if h == 0 or w == 0:
        return None

    # Scale so the longer side fits DIGIT_BOX, preserving aspect ratio
    scale = DIGIT_BOX / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8)
    x_off = (MNIST_SIZE - new_w) // 2
    y_off = (MNIST_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit_resized

    # cv2.imshow("digit",canvas)
    # cv2.waitKey(0)
    return canvas



def sudoku_to_mnist(img_path, include_empty=False):
    """Extract per-cell MNIST-format images from a sudoku image.

    When include_empty=True, empty cells (label 0) are returned as all-black
    28x28 images, representing the absence of a digit.
    """
    try:
        label_data = parse_dat_file(img_path.replace(".jpg", ".dat"))
        matrix = label_data["matrix"]
        result = process_image(img_path, show=False)

    except Exception as e:
        raise ValueError(f"Failed to process {img_path}: {e}")

    X, y, meta = [], [], []
    cells = result.cells   # row-major, 81 entries
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, 9)
        digit = matrix[r][c]

        if digit == 0:
            if not include_empty:
                continue
            # Empty cell → blank image (no digit pixels in MNIST convention)
            X.append(np.zeros((28, 28), dtype=np.uint8))
            y.append(0)
            meta.append({"image_path": img_path, "row": r, "col": c})
            continue

        mnist_cell = cell_to_mnist_format(cell)
        if mnist_cell is None:
            continue

        X.append(mnist_cell)
        y.append(digit)
        meta.append({"image_path": img_path, "row": r, "col": c})

    return X, y, meta
