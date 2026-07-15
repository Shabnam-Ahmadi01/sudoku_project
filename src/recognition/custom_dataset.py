import os
import glob
import numpy as np

from .sudoku_preprocess import sudoku_to_mnist
import sys
import cv2
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.recognition.sudoku_preprocess import cell_to_model_format
import sys

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

def load_chars74():
    """
    Loads Chars74K dataset where each sample folder (sampleXXX) contains images of the same character.
    The folder name indicates the character class (e.g., sample002 = character '1').
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chars74k")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Chars74K data directory not found at {data_dir}. "
                               f"Please ensure the dataset is in the correct location.")
    
    images = []
    labels = []
    
    # Get all sample folders
    sample_folders = sorted([f for f in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, f)) and f.lower().startswith('sample')
                            and 1 <= int(f.lower().replace('sample', '')) <= 10])
    
    print(f"Found {len(sample_folders)} sample folders")
    
    # Map sample folder to class label
    # Assuming sampleXXX format where XXX is the character class
    # For digits: sample002 -> digit '2'
    # For letters: sample010 -> 'A' or 'a' depending on case
    for folder in sample_folders:
        folder_path = os.path.join(data_dir, folder)
        
        # Extract class number from folder name (remove 'sample' prefix)
        class_str = folder.lower().replace('sample', '')
        try:
            class_idx = int(class_str) - 1   # sample001->0 ... sample010->9
            # Class 0 is reserved for empty sudoku cells; skip digit '0' (sample001)
            if class_idx == 0:
                continue
        except ValueError:
            # If it's not a pure number, we need to map it differently
            # For letters, sample010 might be 'A' (or 'a')
            # Map to 0-61 for 62 classes (10 digits + 26 uppercase + 26 lowercase)
            # This is a simplified mapping - you may need to adjust based on your actual dataset
            if len(class_str) == 1:
                if class_str.isdigit():
                    class_idx = int(class_str)
                elif class_str.isupper():
                    class_idx = 10 + ord(class_str) - ord('A')
                elif class_str.islower():
                    class_idx = 36 + ord(class_str) - ord('a')
                else:
                    continue
            else:
                # Handle multi-character class names if needed
                continue
        
        # Load all images in this folder
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(folder_path, img_file)
                raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    # Match inference-time preprocessing exactly: crop to
                    # digit bounding box, invert to white-on-black, center
                    # in a padded 28x28 canvas (same as cell_to_mnist_format
                    # applies to real sudoku cell crops at inference time).
                    img = cell_to_model_format(raw)
                    if img is not None:
                        images.append(img)
                        labels.append(class_idx)
    
    if len(images) == 0:
        raise ValueError("No images found in the dataset. Please check the data directory structure.")
    
    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images from {len(sample_folders)} classes")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split into train and test sets (80-20 split)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return (x_train, y_train), (x_test, y_test)

def load_sudoku_data(empty_fraction=0.15):
    """Load sudoku cell images (classes 1-9 from .dat labels, class 0 = empty cells).
    Returns ((x_train, y_train), (x_test, y_test)) or None if data not found.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "v2_train", "v2_train")
    if not os.path.exists(data_dir):
        print(f"[sudoku data] Directory not found: {data_dir}  — skipping.")
        return None

    X, y, _ = build_custom_digit_dataset(data_dir, empty_fraction=empty_fraction)
    if len(X) == 0:
        print("[sudoku data] No images loaded — skipping.")
        return None

    # Stratify only when every class has enough samples for both splits
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = bool(np.all(counts >= 2))
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None,
    )
    return (x_train, y_train), (x_test, y_test)

# (x_train, y_train), (x_test, y_test) = load_chars74()
# print(f"Chars74K: {len(x_train)} train / {len(x_test)} test samples")

# sudoku_split = load_sudoku_data(empty_fraction=1)
# if sudoku_split is not None:
#     (xs_train, ys_train), (xs_test, ys_test) = sudoku_split
#     print(f"Sudoku:   {len(xs_train)} train / {len(xs_test)} test samples")
#     x_train = np.concatenate([x_train, xs_train])
#     y_train = np.concatenate([y_train, ys_train])
#     x_test  = np.concatenate([x_test,  xs_test])
#     y_test  = np.concatenate([y_test,  ys_test])
#     perm = np.random.permutation(len(x_train))
#     x_train, y_train = x_train[perm], y_train[perm]

# print(f"Combined: {len(x_train)} train / {len(x_test)} test samples")
# print(f"Class distribution (train): {np.bincount(y_train.astype(int))}")

# val_frac = 0.1
# n_val = int(len(x_train) * val_frac)
# x_val, y_val = x_train[:n_val], y_train[:n_val]
# x_train, y_train = x_train[n_val:], y_train[n_val:]

