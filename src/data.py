"""
Data loading, train/val/test splitting, and augmentation for LungCNET.

Expects a directory structure of:

    data/
      benign/     *.png / *.jpg
      malignant/
      normal/

The paper (Section 3.3) splits the dataset 80% for training+validation and
20% for testing, and (Section 3.3 / 4.1) applies data augmentation --
rotation, zoom, horizontal flip, and shift -- via Keras' ImageDataGenerator
to correct class imbalance and improve generalisation. The exact split
between train and validation within the 80% portion, and the exact
augmentation magnitudes, are not stated in the paper; both are configurable
in configs/config.yaml (see README for the assumed defaults).
"""

import pathlib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def _load_image_paths_and_labels(data_dir: str, class_names: list):
    """Walk data_dir/<class_name>/* and return (paths, labels)."""
    data_dir = pathlib.Path(data_dir)
    paths, labels = [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected a folder at {class_dir} -- create it and add the "
                f"'{class_name}' images from the IQ-OTH/NCCD dataset."
            )
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                paths.append(str(img_path))
                labels.append(class_idx)

    if not paths:
        raise FileNotFoundError(
            f"No images found under {data_dir}. Populate data/benign, "
            f"data/malignant, data/normal with the IQ-OTH/NCCD images first."
        )
    return paths, labels


def load_and_split_dataset(cfg: dict):
    """
    Load image paths/labels from cfg['data']['data_dir'], split into
    train/val/test, and return arrays ready for model.fit().

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights_dict
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    class_names = model_cfg["class_names"]
    img_size = tuple(data_cfg["image_size"])
    channels = data_cfg["channels"]
    seed = data_cfg["seed"]

    paths, labels = _load_image_paths_and_labels(data_cfg["data_dir"], class_names)
    labels = np.array(labels)

    # 80% train+val / 20% test, stratified by class (Section 3.3).
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels,
        test_size=data_cfg["test_split"],
        stratify=labels,
        random_state=seed,
    )

    # Split the remaining 80% into train/val.
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=data_cfg["val_split_within_train"],
        stratify=train_val_labels,
        random_state=seed,
    )

    def _load_array(path_list):
        color_mode = "rgb" if channels == 3 else "grayscale"
        imgs = np.zeros((len(path_list), img_size[0], img_size[1], channels), dtype=np.float32)
        for i, p in enumerate(path_list):
            img = load_img(p, target_size=img_size, color_mode=color_mode)
            imgs[i] = img_to_array(img) / 255.0
        return imgs

    X_train = _load_array(train_paths)
    X_val = _load_array(val_paths)
    X_test = _load_array(test_paths)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)

    # Class weighting to correct the class imbalance described in Section 3.3
    # / Figure 5 (Experiment 3), as an alternative/complement to augmentation.
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights_dict = {i: w for i, w in enumerate(weights)}

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights_dict


def build_augmentation_generator(cfg: dict) -> ImageDataGenerator:
    """Build the Keras ImageDataGenerator used for the training set only."""
    aug_cfg = cfg["augmentation"]
    return ImageDataGenerator(
        rotation_range=aug_cfg["rotation_range"],
        zoom_range=aug_cfg["zoom_range"],
        width_shift_range=aug_cfg["width_shift_range"],
        height_shift_range=aug_cfg["height_shift_range"],
        horizontal_flip=aug_cfg["horizontal_flip"],
        fill_mode=aug_cfg["fill_mode"],
    )
