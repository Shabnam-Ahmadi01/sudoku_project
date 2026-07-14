import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import tensorflow as tf
from config import MODEL_PATH
import onnxruntime as ort
from src.recognition.model import (  # noqa: F401
    RandomGaussianBlur, RandomMotionBlur, RandomPerspectiveTransform,
    RandomErodeDilate, RandomJPEGCompression,
)

from tensorflow import keras
import tf2onnx



INPUT_SHAPE = (28, 28, 1)       # grayscale images
BATCH_SIZE = 1
NUM_RUNS = 1000
WARMUP = 20

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -------------------------
# Load Keras model
# -------------------------
print("Loading Keras model...")
keras_model = keras.models.load_model(MODEL_PATH)

# # -------------------------
# # Convert to ONNX
# # -------------------------
# if not os.path.exists(ONNX_MODEL):
#     print("Converting to ONNX...")

#     spec = (
#         tf.TensorSpec(
#             (None, *INPUT_SHAPE),
#             tf.float32,
#             name="input",
#         ),
#     )

#     tf2onnx.convert.from_keras(
#         keras_model,
#         input_signature=spec,
#         output_path=ONNX_MODEL,
#     )

#     print("Saved:", ONNX_MODEL)

# -------------------------
# Load ONNX
# -------------------------
print("Loading ONNX model...")
session = ort.InferenceSession(
    MODEL_PATH.replace(".keras", ".onnx"),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# -------------------------
# Dummy input
# -------------------------
dummy = np.random.rand(
    BATCH_SIZE,
    *INPUT_SHAPE
).astype(np.float32)

# -------------------------
# Compare outputs
# -------------------------
keras_output = keras_model(dummy, training=False).numpy()
onnx_output = session.run(None, {input_name: dummy})[0]

max_diff = np.max(np.abs(keras_output - onnx_output))

print("\nOutput Verification")
print("-------------------")
print("Maximum difference:", max_diff)

# -------------------------
# Benchmark Keras
# -------------------------
for _ in range(WARMUP):
    keras_model(dummy, training=False)

start = time.perf_counter()

for _ in range(NUM_RUNS):
    keras_model(dummy, training=False)

keras_time = time.perf_counter() - start

keras_ms = keras_time / NUM_RUNS * 1000

# -------------------------
# Benchmark ONNX
# -------------------------
for _ in range(WARMUP):
    session.run(None, {input_name: dummy})

start = time.perf_counter()

for _ in range(NUM_RUNS):
    session.run(None, {input_name: dummy})

onnx_time = time.perf_counter() - start

onnx_ms = onnx_time / NUM_RUNS * 1000

# -------------------------
# Results
# -------------------------
print("\nBenchmark")
print("-------------------------------------------")
print(f"Keras CPU : {keras_ms:.3f} ms/image")
print(f"ONNX  CPU : {onnx_ms:.3f} ms/image")
print(f"Speedup   : {keras_ms / onnx_ms:.2f}x")
print("-------------------------------------------")