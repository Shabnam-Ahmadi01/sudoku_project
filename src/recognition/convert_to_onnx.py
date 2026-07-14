import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow as tf
import tf2onnx
from tensorflow import keras
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import MODEL_PATH
from src.recognition.model import (  # noqa: F401
    RandomGaussianBlur, RandomMotionBlur, RandomPerspectiveTransform,
    RandomErodeDilate, RandomJPEGCompression,
)

model = keras.models.load_model(MODEL_PATH)

spec = (
    tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),
)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path="model.onnx",
)

print("Done!")