from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

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


@keras.utils.register_keras_serializable(package="sudoku_recognition")
class RandomMotionBlur(layers.Layer):

    def __init__(self, probability=0.3, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.kernel_size = kernel_size

    def call(self, x, training=None):
        if not training:
            return x

        n = self.kernel_size
        mid = n // 2

        # Build all 4 direction kernels as constants (no runtime branching on numpy)
        h = np.zeros((n, n), np.float32); h[mid, :] = 1; h /= h.sum()  # horizontal
        v = np.zeros((n, n), np.float32); v[:, mid] = 1; v /= v.sum()  # vertical
        d = np.eye(n, dtype=np.float32); d /= d.sum()                  # diagonal
        a = np.fliplr(d.copy()); a /= a.sum()                           # anti-diagonal

        kernels = tf.constant(
            np.stack([h, v, d, a])[:, :, :, np.newaxis, np.newaxis],   # (4, n, n, 1, 1)
            dtype=tf.float32,
        )

        direction = tf.random.uniform([], 0, 4, dtype=tf.int32)
        kernel = kernels[direction]                                       # (n, n, 1, 1)
        channels = tf.shape(x)[-1]
        kernel = tf.tile(kernel, tf.stack([1, 1, channels, 1]))

        blurred = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.cond(tf.random.uniform([]) < self.probability, lambda: blurred, lambda: x)


@keras.utils.register_keras_serializable(package="sudoku_recognition")
class RandomPerspectiveTransform(layers.Layer):
    """Random projective (perspective) warp for simulating camera angle changes."""

    def __init__(self, magnitude=1.5, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.magnitude = magnitude
        self.probability = probability

    def call(self, x, training=None):
        if not training:
            return x

        batch_size = tf.shape(x)[0]
        m = self.magnitude
        ones = tf.ones([batch_size])
        # 8-param inverse mapping: x_src = (a0*xd + a1*yd + a2) / (c0*xd + c1*yd + 1)
        a0 = ones + tf.random.uniform([batch_size], -0.05 * m, 0.05 * m)
        a1 = tf.random.uniform([batch_size], -0.1 * m, 0.1 * m)
        a2 = tf.random.uniform([batch_size], -m, m)
        b0 = tf.random.uniform([batch_size], -0.1 * m, 0.1 * m)
        b1 = ones + tf.random.uniform([batch_size], -0.05 * m, 0.05 * m)
        b2 = tf.random.uniform([batch_size], -m, m)
        c0 = tf.random.uniform([batch_size], -0.003 * m, 0.003 * m)
        c1 = tf.random.uniform([batch_size], -0.003 * m, 0.003 * m)
        transforms = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1], axis=1)

        warped = tf.raw_ops.ImageProjectiveTransformV3(
            images=x,
            transforms=transforms,
            output_shape=tf.shape(x)[1:3],
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
            fill_value=0.0,
        )
        return tf.cond(tf.random.uniform([]) < self.probability, lambda: warped, lambda: x)


@keras.utils.register_keras_serializable(package="sudoku_recognition")
class RandomErodeDilate(layers.Layer):
    """Randomly erodes or dilates strokes to simulate pen thickness variation."""

    def __init__(self, kernel_size=2, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.probability = probability

    def call(self, x, training=None):
        if not training:
            return x

        k = self.kernel_size
        depth = tf.shape(x)[-1]
        # Flat (all-zero) structuring element: pure shape-based min/max filter
        se = tf.zeros(tf.stack([k, k, depth]), dtype=tf.float32)

        eroded = tf.nn.erosion2d(
            x, se, strides=[1, 1, 1, 1], padding="SAME",
            data_format="NHWC", dilations=[1, 1, 1, 1],
        )
        dilated = tf.nn.dilation2d(
            x, se, strides=[1, 1, 1, 1], padding="SAME",
            data_format="NHWC", dilations=[1, 1, 1, 1],
        )
        morphed = tf.cond(tf.random.uniform([]) < 0.5, lambda: dilated, lambda: eroded)
        return tf.cond(tf.random.uniform([]) < self.probability, lambda: morphed, lambda: x)


@keras.utils.register_keras_serializable(package="sudoku_recognition")
class RandomJPEGCompression(layers.Layer):
    """Applies random JPEG compression to simulate camera/scan artifacts."""

    def __init__(self, min_quality=20, max_quality=80, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.probability = probability

    def _compress_numpy(self, images):
        import cv2
        results = []
        for img in images:
            quality = int(np.random.randint(self.min_quality, self.max_quality + 1))
            img_u8 = np.clip(img, 0, 255).astype(np.uint8).squeeze(-1)
            _, enc = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
            dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            results.append(dec[..., np.newaxis].astype(np.float32))
        return np.array(results)

    def call(self, x, training=None):
        if not training:
            return x

        def compress():
            out = tf.py_function(self._compress_numpy, [x], tf.float32)
            out.set_shape(x.shape)
            return out

        return tf.cond(tf.random.uniform([]) < self.probability, compress, lambda: x)


def build_augmentation():
    """Augmentation for printed/handwritten digits from camera or scan.
    Kept conservative: large rotations would turn a 6 into a 9."""
    return keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.08),
        RandomPerspectiveTransform(magnitude=1.5, probability=0.4),
        RandomGaussianBlur(min_sigma=0.0, max_sigma=2.0, kernel_size=7),
        RandomMotionBlur(probability=0.3, kernel_size=5),
        RandomErodeDilate(kernel_size=2, probability=0.4),
        # RandomJPEGCompression(min_quality=25, max_quality=85, probability=0.3),
    ], name="augmentation")
