import numpy as np
from tensorflow import keras
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .sudoku_preprocess import sudoku_to_mnist
from config import MODEL_PATH


def predict_matrix(img_path, confidence_threshold=0.6):
    X, y, meta = sudoku_to_mnist(img_path)
    X = np.array(X, dtype=np.uint8)[..., np.newaxis]
    y = np.array(y, dtype=np.int64)

    model = keras.models.load_model(MODEL_PATH)
    predictions = model.predict(X, verbose=0)

    y_pred = np.argmax(predictions, axis=1)
    confidences = predictions[np.arange(len(y_pred)), y_pred]  # max prob per cell

    sudoku_matrix = np.zeros((9, 9), dtype=int)
    for i, r in enumerate(meta):
        if confidences[i] < confidence_threshold:
            continue  # stays 0 ("empty") -- low confidence, don't risk a wrong digit
        sudoku_matrix[r["row"], r["col"]] = int(y_pred[i])

    return sudoku_matrix, confidences
