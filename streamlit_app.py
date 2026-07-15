"""
Simple Streamlit UI for the sudoku_project pipeline.

Shows: input image -> extracted (recognized) grid -> final solved answer.

Run from the PROJECT ROOT (same folder as config.py):
    pip install streamlit onnxruntime
    streamlit run streamlit_app.py

Why ONNX and not the Keras model: only models/chars74_cnn.onnx is committed
to the repo (models/*.keras is gitignored), so this app loads that directly
with onnxruntime. No TensorFlow install needed. If you later add back the
.keras file and want to use it instead, swap the ModelRunner implementation
below -- everything else (grid detection, cell extraction, solving, UI)
stays the same.
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIGIT_BOX, MNIST_SIZE, WARP_SIZE
from src.preprocessing import process_image
from src.solver import solve_sudoku

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "chars74_cnn.onnx")


# ---------------------------------------------------------------------------
# Recognition (re-implemented against ONNX directly, to avoid importing
# src.recognition, which pulls in `from tensorflow import keras` at package
# import time even though inference itself doesn't need TensorFlow here).
# ---------------------------------------------------------------------------

def cell_to_mnist_format(cell_gray):
    """Same logic as src/recognition/sudoku_preprocess.py::cell_to_mnist_format."""
    if cell_gray is None or cell_gray.size == 0:
        return None
    blur = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    digit = binary[y1:y2 + 1, x1:x2 + 1]
    h, w = digit.shape
    if h == 0 or w == 0:
        return None
    scale = DIGIT_BOX / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8)
    x_off, y_off = (MNIST_SIZE - new_w) // 2, (MNIST_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit_resized
    return canvas


@st.cache_resource
def load_onnx_session(model_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def predict_from_cells_onnx(sess, input_name, cells, empty_mask, confidence_threshold=0.0):
    """Mirrors src/recognition/__init__.py::predict_from_cells, but calls the
    ONNX session instead of a Keras model."""
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

    X_arr = np.array(X, dtype=np.float32)[..., np.newaxis]
    predictions = sess.run(None, {input_name: X_arr})[0]
    y_pred = np.argmax(predictions, axis=1)
    confidences = predictions[np.arange(len(y_pred)), y_pred]

    matrix = np.zeros((9, 9), dtype=int)
    for i, (r, c) in enumerate(meta):
        if confidences[i] >= confidence_threshold:
            matrix[r, c] = int(y_pred[i])
    return matrix, confidences


# ---------------------------------------------------------------------------
# Full pipeline: same 4-rotation retry strategy as src/api/app.py, so results
# match the API's behavior.
# ---------------------------------------------------------------------------

def run_pipeline(image_path, sess, input_name, confidence_threshold=0.0):
    process_result = process_image(image_path)
    warped_gray = process_result.warped_gray

    first_failure = None
    for k in range(4):
        rotated = np.rot90(warped_gray, k=k)

        from src.preprocessing.cell_extractor import split_into_cells, is_cell_empty
        cells = split_into_cells(rotated)
        empty_mask = [is_cell_empty(c) for c in cells]

        recognized, confidences = predict_from_cells_onnx(
            sess, input_name, cells, empty_mask, confidence_threshold
        )

        try:
            solved, conflicts = solve_sudoku(recognized, raise_on_invalid=False)
        except ValueError:
            # board consistent but no solution exists for this rotation
            solved, conflicts = None, []

        if not conflicts and solved is not None:
            unrot = (4 - k) % 4
            rec_out = np.rot90(recognized, k=unrot)
            sol_out = np.rot90(np.array(solved), k=unrot)
            return {
                "status": "success",
                "recognized": rec_out,
                "solved": sol_out,
                "warped": np.rot90(warped_gray, k=0),  # display in original orientation
                "mean_confidence": float(confidences.mean()) if len(confidences) else 0.0,
                "rotation": k,
            }

        if first_failure is None:
            first_failure = {
                "recognized": recognized,
                "confidences": confidences,
                "conflicts": conflicts,
            }

    conf = first_failure["confidences"]
    mean_conf = float(conf.mean()) if len(conf) else 0.0
    return {
        "status": "recognition_invalid" if first_failure["conflicts"] else "unsolvable",
        "recognized": first_failure["recognized"],
        "solved": None,
        "warped": warped_gray,
        "mean_confidence": mean_conf,
        "conflicts": first_failure["conflicts"],
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_grid_html(matrix, given_mask=None, highlight_color="#1f6feb"):
    """Renders a 9x9 matrix as a bordered HTML sudoku grid.
    given_mask: optional 9x9 bool array -- True cells are drawn bold/black
    (originally recognized), False cells drawn in highlight_color (solver-filled).
    """
    cells_html = []
    for r in range(9):
        for c in range(9):
            v = matrix[r][c]
            text = str(v) if v != 0 else ""
            is_given = True if given_mask is None else bool(given_mask[r][c])
            color = "#e6e6e6" if is_given else highlight_color
            weight = "700" if is_given else "600"

            border_top = "2px solid #888" if r % 3 == 0 else "1px solid #444"
            border_left = "2px solid #888" if c % 3 == 0 else "1px solid #444"
            border_right = "2px solid #888" if c == 8 else "none"
            border_bottom = "2px solid #888" if r == 8 else "none"

            style = (
                f"width:38px;height:38px;text-align:center;vertical-align:middle;"
                f"font-size:18px;font-weight:{weight};color:{color};"
                f"border-top:{border_top};border-left:{border_left};"
                f"border-right:{border_right};border-bottom:{border_bottom};"
            )
            cells_html.append(f'<td style="{style}">{text}</td>')

    rows = "".join(
        f"<tr>{''.join(cells_html[r * 9:(r + 1) * 9])}</tr>" for r in range(9)
    )
    return f'<table style="border-collapse:collapse;margin:auto;">{rows}</table>'


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Sudoku Solver", layout="wide")
st.title("Sudoku Recognizer & Solver")
st.caption("Upload a photo of a sudoku grid to recognize and solve it.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Make sure "
             f"models/chars74_cnn.onnx exists in the project.")
    st.stop()

sess, input_name = load_onnx_session(MODEL_PATH)

uploaded = st.file_uploader("Upload sudoku image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        image_path = tmp.name

    try:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Input image")
            st.image(uploaded, use_container_width=True)

        with st.spinner("Detecting grid, recognizing digits, solving..."):
            try:
                result = run_pipeline(image_path, sess, input_name)
            except ValueError as e:
                result = {"status": "grid_detection_failed", "error": str(e)}

        if result["status"] == "grid_detection_failed":
            st.error(f"Could not detect a sudoku grid in this image: "
                      f"{result.get('error', '')}")

        else:
            with col1:
                st.image(result["warped"], caption="Rectified grid", use_container_width=True,
                          channels="GRAY")

            with col2:
                st.subheader("Extracted grid")
                st.markdown(render_grid_html(result["recognized"]), unsafe_allow_html=True)
                st.caption(f"Mean recognition confidence: {result['mean_confidence']:.2f}")

            with col3:
                st.subheader("Final answer")
                if result["status"] == "success":
                    given_mask = (result["recognized"] != 0)
                    st.markdown(render_grid_html(result["solved"], given_mask),
                                unsafe_allow_html=True)
                    st.success(f"Solved (orientation attempt {result['rotation']}/4).")
                elif result["status"] == "recognition_invalid":
                    st.warning("Recognized digits conflict with sudoku rules "
                               "(likely a misread digit) -- no solution shown.")
                    st.write("Conflicting cells:", result["conflicts"])
                else:
                    st.warning("Recognized grid is valid but has no solution "
                               "-- likely a misread digit.")

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
else:
    st.info("Waiting for an image upload.")
