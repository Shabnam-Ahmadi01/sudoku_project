import os
import numpy as np
import matplotlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow import keras

from .custom_dataset import build_custom_digit_dataset
from config import OUT_DIR, MODEL_PATH, DATA_PATH,TEST_DATA_PATH

def evaluate(model, X, y, tag="custom_data"):
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    acc = (y_pred == y).mean()
    print(f"\nAccuracy on {tag}: {acc:.4f}  ({len(y)} samples)")
    print("\nClassification report (true labels are only 1-9; class 0 "
          "column shows false predictions of 'no digit'):\n")
    print(classification_report(y, y_pred, labels=list(range(10)),
                                 zero_division=0, digits=3))

    cm = confusion_matrix(y, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title(f"MNIST model on {tag} (acc={acc:.3f})")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{tag}_confusion_matrix.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")
    # print(y_pred)

    return acc, cm, y_pred


def main(max_images=None, data_path=DATA_PATH):

    model = keras.models.load_model(MODEL_PATH)

    X, y, meta = build_custom_digit_dataset(data_path, max_images=max_images)

    if len(X) == 0:
        print("No non-empty cells collected -- check data_root path.")
        return

    _,_,y_pred = evaluate(model, X, y, tag="sudoku_nonempty_cells")
    return y_pred,meta

if __name__ == "__main__":
  
    main(40, data_path=TEST_DATA_PATH)