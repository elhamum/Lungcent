"""
LungCNET architecture.

Three convolutional blocks (32 -> 64 -> 128 filters, ReLU, MaxPooling2D after
each), flattened into two dense layers (512, 256) each followed by dropout at
rate 0.5, ending in a softmax output over three classes: benign, malignant,
normal. Trained from scratch (no transfer learning), per Section 3.2 of the
paper.
"""

from tensorflow.keras import layers, models


def build_lungcnet(
    input_shape: tuple = (224, 224, 3),
    conv_filters: tuple = (32, 64, 128),
    kernel_size: tuple = (3, 3),
    pool_size: tuple = (2, 2),
    dense_units: tuple = (512, 256),
    dropout_rate: float = 0.5,
    num_classes: int = 3,
) -> models.Model:
    """Build and return an uncompiled LungCNET Keras model."""
    inputs = layers.Input(shape=input_shape, name="ct_scan_input")
    x = inputs

    for i, n_filters in enumerate(conv_filters):
        x = layers.Conv2D(
            n_filters, kernel_size, activation="relu", padding="same",
            name=f"conv_block_{i + 1}",
        )(x)
        x = layers.MaxPooling2D(pool_size, name=f"maxpool_{i + 1}")(x)

    x = layers.Flatten(name="flatten")(x)

    for i, n_units in enumerate(dense_units):
        x = layers.Dense(n_units, activation="relu", name=f"dense_{i + 1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i + 1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LungCNET")
    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and print a summary.
    model = build_lungcnet()
    model.summary()
