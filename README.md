# LungCNET

A custom convolutional neural network for classifying lung CT scans into **benign**,
**malignant**, and **normal**, trained on the [IQ-OTH/NCCD Lung Cancer Dataset](https://doi.org/10.17632/bhmdr45bh2.2).

This implementation follows the architecture and training procedure described in:

> Eskandarnia, E., Adepoju, P., Kiani, K., Mansouri, T., BinRajab, A.
> *LungCNET: A High-Performance Deep CNN Model for Lung Cancer Detection Evaluated
> Against Widely Used CNN Benchmarks.* Submitted to *Bioengineering* (MDPI), 2026.

## Architecture

Three convolutional blocks, followed by two dense layers with dropout, ending in a
3-class softmax output:

```
Input (224x224x3)
 -> Conv2D(32, 3x3, ReLU) -> MaxPooling2D(2x2)
 -> Conv2D(64, 3x3, ReLU) -> MaxPooling2D(2x2)
 -> Conv2D(128, 3x3, ReLU) -> MaxPooling2D(2x2)
 -> Flatten
 -> Dense(512, ReLU) -> Dropout(0.5)
 -> Dense(256, ReLU) -> Dropout(0.5)
 -> Dense(3, Softmax)
```

Trained from scratch (no transfer learning), with the Adam optimizer and categorical
cross-entropy loss.

## Parameter used 

The Methodology section describes the architecture and training procedure, we used this values for hyperparameter

| Parameter | Value used here | Where it's set |
|---|---|---|
| Learning rate | `1e-4` | `configs/config.yaml` |
| Batch size | `32` | `configs/config.yaml` |
| Max epochs | `100` (early stopping will cut this short — paper reports convergence at 18 epochs) | `configs/config.yaml` |
| Early stopping patience | `10` epochs on validation loss | `configs/config.yaml` |
| Augmentation ranges (rotation, zoom, shift) | rotation ±20°, zoom 0.2, width/height shift 0.2, horizontal flip | `configs/config.yaml` |
| Train/val split within the 80% training portion | 80/20 (i.e. 64% train / 16% val / 20% test overall) | `configs/config.yaml` |
| Class imbalance correction (Experiment 3) | class weighting via `sklearn.utils.class_weight`, applied at `model.fit()` | `src/train.py` |

## Project structure

```
lungcnet/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── benign/       <- put IQ-OTH/NCCD benign images here
│   ├── malignant/    <- put IQ-OTH/NCCD malignant images here
│   ├── normal/       <- put IQ-OTH/NCCD normal images here
│   └── augmented/    <- generated automatically when you run training;
│                        not something you populate by hand, and not
│                        something we distribute -- see note below
└── src/
    ├── model.py       # LungCNET architecture
    ├── data.py        # data loading, splitting, augmentation
    ├── train.py        # training loop with callbacks
    └── evaluate.py      # per-class precision/recall/F1, confusion matrix
```

## Setup

```bash
git clone <your-repo-url>
cd lungcnet
pip install -r requirements.txt
```

Download the [IQ-OTH/NCCD dataset](https://doi.org/10.17632/bhmdr45bh2.2) and place
images into `data/benign/`, `data/malignant/`, `data/normal/` (one folder per class,
matching Keras' `flow_from_directory` convention).

## Training

```bash
python -m src.train --config configs/config.yaml
```

This will:
1. Split the dataset 80% train+validation / 20% test (stratified by class).
2. Apply data augmentation (rotation, zoom, horizontal flip, shift) to the training set only.
3. Train LungCNET from scratch with early stopping and TensorBoard logging.
4. Save the best model checkpoint to `checkpoints/lungcnet_best.h5`.

Monitor training with:

```bash
tensorboard --logdir runs/
```

## Evaluation

```bash
python -m src.evaluate --model checkpoints/lungcnet_best.h5 --config configs/config.yaml
```

Prints per-class precision/recall/F1, the macro-averaged F1-score, and saves a
confusion matrix plot to `outputs/confusion_matrix.png`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff` or the reference above).

## License

MIT (adjust as needed for your submission/institutional requirements).
