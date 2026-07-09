"""
Train LungCNET from scratch with data augmentation, early stopping, and
TensorBoard logging, per Section 3.2 / 3.3 of the paper.

Usage:
    python -m src.train --config configs/config.yaml
"""

import argparse
import datetime
import pathlib

import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from src.data import build_augmentation_generator, load_and_split_dataset
from src.model import build_lungcnet


def parse_args():
    parser = argparse.ArgumentParser(description="Train LungCNET.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    print("Loading and splitting dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights = \
        load_and_split_dataset(cfg)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Class weights (imbalance correction): {class_weights}")

    y_train_cat = to_categorical(y_train, num_classes=model_cfg["num_classes"])
    y_val_cat = to_categorical(y_val, num_classes=model_cfg["num_classes"])

    input_shape = (*data_cfg["image_size"], data_cfg["channels"])
    model = build_lungcnet(
        input_shape=input_shape,
        conv_filters=tuple(model_cfg["conv_filters"]),
        kernel_size=tuple(model_cfg["kernel_size"]),
        pool_size=tuple(model_cfg["pool_size"]),
        dense_units=tuple(model_cfg["dense_units"]),
        dropout_rate=model_cfg["dropout_rate"],
        num_classes=model_cfg["num_classes"],
    )
    model.compile(
        optimizer=Adam(learning_rate=train_cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint_dir = pathlib.Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(train_cfg["log_dir"]) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_dir / "lungcnet_best.h5"),
            monitor="val_loss",
            save_best_only=True,
        ),
        TensorBoard(log_dir=str(log_dir)),
    ]

    datagen = build_augmentation_generator(cfg)
    datagen.fit(X_train)
    train_flow = datagen.flow(
        X_train, y_train_cat, batch_size=train_cfg["batch_size"],
    )

    fit_kwargs = dict(
        validation_data=(X_val, y_val_cat),
        epochs=train_cfg["max_epochs"],
        callbacks=callbacks,
    )
    if train_cfg.get("use_class_weights", False):
        fit_kwargs["class_weight"] = class_weights

    print("Starting training...")
    model.fit(train_flow, **fit_kwargs)

    final_path = checkpoint_dir / "lungcnet_final.h5"
    model.save(final_path)
    print(f"Training complete. Best checkpoint: {checkpoint_dir / 'lungcnet_best.h5'}")
    print(f"Final model saved to: {final_path}")
    print(f"TensorBoard logs: {log_dir}  (run: tensorboard --logdir {train_cfg['log_dir']})")


if __name__ == "__main__":
    main()
