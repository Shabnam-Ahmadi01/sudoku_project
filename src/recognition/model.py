from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation():
    """Light augmentation appropriate for printed/handwritten digits.
    Kept conservative: e.g. large rotations would turn a 6 into a 9."""
    return keras.Sequential([
        layers.RandomRotation(0.06),          # ~ +/-10 degrees
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.08),
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