from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

@keras.utils.register_keras_serializable(package="sudoku_recognition")
class RandomGaussianBlur(layers.Layer):
    """Applies a random-sigma Gaussian blur, only at training time."""
    def __init__(self, min_sigma=0.0, max_sigma=2.0, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.kernel_size = kernel_size

    def _gaussian_kernel(self, sigma):
        ax = tf.range(-(self.kernel_size // 2), self.kernel_size // 2 + 1, dtype=tf.float32)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel[:, :, tf.newaxis, tf.newaxis]

    def call(self, x, training=None):
        if not training:
            return x
        sigma = tf.random.uniform([], self.min_sigma, self.max_sigma)
        kernel = self._gaussian_kernel(sigma)
        channels = tf.shape(x)[-1]
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")


def build_augmentation():
    """Light augmentation appropriate for printed/handwritten digits.
    Kept conservative: e.g. large rotations would turn a 6 into a 9."""
    return keras.Sequential([
        layers.RandomRotation(0.06),          # ~ +/-10 degrees
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.08),
        RandomGaussianBlur(min_sigma=0.0, max_sigma=2.0, kernel_size=7),
    ], name="augmentation")


def build_cnn(input_shape=(28, 28, 1), num_classes=10, augment=True):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    if augment:
        x = build_augmentation()(x)

    # normalize 0-255 -> 0-1 inside the model so the same saved model can
    # accept raw uint8-range input consistently at inference time too
    x = layers.Rescaling(1.0 / 255)(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="digit_cnn")
    return model