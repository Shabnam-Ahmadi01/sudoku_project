import numpy as np
from tensorflow import keras
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .sudoku_preprocess import sudoku_to_mnist, cell_to_mnist_format
from config import CONFIDENCE_THRESHOLD


def predict_from_cells(model, cells, empty_mask, confidence_threshold=CONFIDENCE_THRESHOLD):
    X, meta = [], []
    for idx, (cell, is_empty) in enumerate(zip(cells, empty_mask)):
        if is_empty:
            continue
        mnist_cell = cell_to_mnist_format(cell)
        if mnist_cell is None:
            continue
        r, c = divmod(idx, 9)
        X.append(mnist_cell)
        meta.append((r, c))

    if not X:
        return np.zeros((9, 9), dtype=int), np.zeros(0)

    X_arr = np.array(X, dtype=np.uint8)[..., np.newaxis]
    predictions = model.predict(X_arr, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    confidences = predictions[np.arange(len(y_pred)), y_pred]

    matrix = np.zeros((9, 9), dtype=int)
    for i, (r, c) in enumerate(meta):
        if confidences[i] >= confidence_threshold:
            matrix[r, c] = int(y_pred[i])

    return matrix, confidences


def predict_matrix(model, img_path, confidence_threshold=CONFIDENCE_THRESHOLD):
    X, _, meta = sudoku_to_mnist(img_path)
    X = np.array(X, dtype=np.uint8)[..., np.newaxis]

    predictions = model.predict(X, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    confidences = predictions[np.arange(len(y_pred)), y_pred]

    sudoku_matrix = np.zeros((9, 9), dtype=int)
    for i, r in enumerate(meta):
        if confidences[i] >= confidence_threshold:
            sudoku_matrix[r["row"], r["col"]] = int(y_pred[i])

    return sudoku_matrix, confidences
