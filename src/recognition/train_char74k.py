"""
Produces:
  models/chars74_cnn.keras         - trained model
  outputs/chars74_confusion_matrix.png
  outputs/chars74_training_curves.png
"""
import os
import numpy as np
import matplotlib
import sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow import keras
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.recognition.custom_dataset import load_chars74,load_sudoku_data
import sys
from model import build_cnn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)



def plot_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def main(epochs=20, batch_size=128):
    (x_train, y_train), (x_test, y_test) = load_chars74()
    print(f"Chars74K: {len(x_train)} train / {len(x_test)} test samples")

    sudoku_split = load_sudoku_data(empty_fraction=0.15)
    if sudoku_split is not None:
        (xs_train, ys_train), (xs_test, ys_test) = sudoku_split
        print(f"Sudoku:   {len(xs_train)} train / {len(xs_test)} test samples")
        x_train = np.concatenate([x_train, xs_train])
        y_train = np.concatenate([y_train, ys_train])
        x_test  = np.concatenate([x_test,  xs_test])
        y_test  = np.concatenate([y_test,  ys_test])
        # Shuffle combined training set
        perm = np.random.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]

    print(f"Combined: {len(x_train)} train / {len(x_test)} test samples")
    print(f"Class distribution (train): {np.bincount(y_train.astype(int))}")

    # held-out validation split from training data
    val_frac = 0.1
    n_val = int(len(x_train) * val_frac)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    # Always 10 classes: 0=empty cell, 1-9=sudoku digits
    num_classes = 10
    print(f"Number of classes: {num_classes}")
    
    model = build_cnn(input_shape=(28, 28, 1), num_classes=num_classes, augment=True)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",   # cross entropy, integer labels
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=2),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_curves(history, os.path.join(OUT_DIR, "chars74_training_curves.png"))

    # final evaluation on held-out test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nChars74K test accuracy: {test_acc:.4f}  (loss: {test_loss:.4f})")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nClassification report:\n",
          classification_report(y_test, y_pred, digits=3))

    # Create labels for all classes
    labels = list(range(num_classes))
    
    plot_confusion_matrix(
        y_test, y_pred, labels=labels,
        path=os.path.join(OUT_DIR, "chars74_confusion_matrix.png"),
        title=f"Chars74K Test Confusion Matrix (acc={test_acc:.3f})",
    )

    model.save(os.path.join(MODEL_DIR, "chars74_cnn.keras"))
    print(f"\nSaved model to {os.path.join(MODEL_DIR, 'chars74_cnn.keras')}")
    return model, history


if __name__ == "__main__":
    main()