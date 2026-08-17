# Multi-Attribute Biometrics Application

PyTorch experiments for multi-attribute visual recognition on **UTKFace** and **PETA**, with a real-time webcam inference demo for the UTKFace model.

The project explores two related tasks:

- **UTKFace:** joint age estimation, gender classification, and race classification.
- **PETA:** pedestrian attribute recognition across binary attributes and clothing/hair/footwear color attributes.

Two model families are implemented: a transfer-learning pipeline based on **MobileNetV3-Small** and a configurable custom CNN. The original experiment script includes hyperparameter search, augmentation, mixed-precision training, learning-rate scheduling, early stopping, and task-specific evaluation.

## Repository structure

```text
.
├── train.py                    # cleaned training/evaluation entry point
├── training.py                 # original full experiment + hyperparameter-search script
├── demo/
│   └── webcamPredictor.py      # real-time UTKFace webcam inference
├── requirements.txt
└── LICENSE
```

`train.py` is the recommended starting point. `training.py` is retained as the original experiment artifact and contains the broader hyperparameter-search workflow.

## Models

### MobileNetV3 transfer learning

The transfer-learning model uses an ImageNet-pretrained MobileNetV3-Small backbone followed by a shared 512-unit representation layer. For UTKFace, separate heads predict age, gender, and race. For PETA, the output layer predicts the complete attribute vector.

### Custom CNN

The configurable CNN supports multiple convolutional depths and hidden-layer sizes, allowing direct comparison with the pretrained backbone approach.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Datasets are intentionally not committed to the repository.

```text
Datasets/
├── PETA/
│   └── TownCentre/
│       ├── Label.txt
│       └── ... images ...
└── UTK/
    └── ... UTKFace .jpg files ...
```

The UTKFace loader expects filenames following the dataset convention `age_gender_race_*.jpg`.

## Training

UTKFace with MobileNetV3:

```bash
python train.py --dataset utkface --model mobilenet --data-root ./Datasets/UTK
```

PETA with the custom CNN:

```bash
python train.py --dataset peta --model mlcnn --data-root ./Datasets/PETA
```

Useful options include:

```text
--epochs
--batch-size
--learning-rate
--weight-decay
--dropout
--seed
--num-workers
```

Models are written to `models/`.

## Evaluation

- **UTKFace:** mean absolute error (MAE) for age, accuracy for gender, and accuracy for race.
- **PETA:** micro-F1 across the predicted attribute vector.

The cleaned entry point keeps validation preprocessing separate from training augmentation and treats the PETA outputs as logits during optimization.

## Webcam demo

`demo/webcamPredictor.py` performs live webcam inference with a trained UTKFace model and overlays age, gender, and race predictions on the video stream. The demo expects a compatible saved model checkpoint.

## Notes

This repository is an academic/research implementation rather than a production biometric system. The demographic labels are inherited from the source datasets and should not be interpreted as a general-purpose or deployment-ready characterization of people. Dataset licenses and terms remain those of the original PETA and UTKFace sources.
