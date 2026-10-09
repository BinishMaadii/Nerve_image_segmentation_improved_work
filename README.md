from pathlib import Path

readme = r"""# Ultrasound Nerve Segmentation with PyTorch and MLflow

A research notebook for segmenting the brachial plexus nerve in ultrasound images. The notebook develops a small U-Net-based model step by step, tracks experiments with MLflow, evaluates performance using patient-level splits, and investigates model explanations.

> **Scope:** This is an experimental machine-learning project, not a clinically validated medical device. Results depend on the dataset, split, training settings, and execution environment.

## What the notebook does

The notebook contains a sequence of experiments:

1. **Baseline segmentation** with a small U-Net.
2. **Oversampling** to show frames containing nerve more frequently during training.
3. **Imbalance-aware loss**, combining weighted pixel-wise loss and a Dice-based term.
4. **Presence detection head**, which predicts whether a frame contains nerve in addition to producing a segmentation mask.
5. **Post-processing and threshold selection**, including detection/pixel thresholds and connected-component filtering.
6. **Cross-validation, model ensembling, and horizontal-flip averaging** experiments.
7. **Patient-level bootstrap confidence intervals** and TorchScript export.
8. **Persistent training pipeline** that saves models, the data cache, patient split, and training history.
9. **Diagnostics** for detection, segmentation, threshold selection, and variation across patients/folds.
10. **Explainability analysis** using Grad-CAM, Head-CAM, Seg-Grad-CAM, and occlusion, with quantitative checks.
11. **Ablation study** comparing successive modelling choices with patient-level uncertainty estimates.

The later steps are designed to reuse the saved artifacts from the persistent training step. The notebook also contains earlier, self-contained iterations of the model; for the reproducible diagnostics workflow, use the persistent step and then run its dependent steps in order.

## Dataset

The code is written for the Kaggle **Ultrasound Nerve Segmentation** competition dataset, containing neck ultrasound images and brachial plexus nerve masks.

Download the dataset from Kaggle and set `DATA_DIR` to the directory containing the image files. The loader searches recursively for image/mask pairs named:

- Image: `<patient>_<frame>.tif`
- Mask: `<patient>_<frame>_mask.tif`

Files without a matching mask are skipped. Images are converted to grayscale and resized to **96 × 128** pixels using bilinear interpolation. Masks are resized with nearest-neighbour interpolation and binarised at intensity `> 127`.

The default `DATA_DIR` in the persistent pipeline is `/kaggle/input/competitions/ultrasound-nerve-segmentation`. Outside Kaggle, set it to the local dataset directory.

## Methods

### Model

The network is a compact U-Net-style convolutional encoder–decoder with skip connections. It produces two outputs:

- **Segmentation logits:** a pixel-wise nerve mask.
- **Presence logit:** a frame-level score for whether nerve is present.

The presence output can help suppress false-positive masks on frames without visible nerve.

### Training and evaluation design

- Inputs are normalised to `[0, 1]`.
- The fixed random seed is `0`.
- Training uses batches of `32` and defaults to `10` epochs.
- The persistent pipeline reserves approximately 20% of patients for a held-out test set using `GroupShuffleSplit`.
- The remaining development patients are split into groups using `GroupKFold` (default: five folds).
- Patient grouping is used to reduce leakage between frames from the same patient.
- The loss gives greater weight to nerve pixels and includes a Dice term; a separate loss term trains the presence head.
- The diagnostics step selects configurations using out-of-fold development predictions and reports test-set results separately.
- Confidence intervals resample patients rather than individual frames, accounting for correlation among frames from the same patient.

The test set is intended for final reporting, not for choosing thresholds or model settings. Small patient counts can produce wide uncertainty intervals; a point estimate alone should not be interpreted as proof of improvement.

### Metrics and diagnostics

Depending on the step, the notebook reports or plots:

- Detection: ROC-AUC, precision–recall analysis, average precision, and calibration.
- Segmentation: Dice, intersection over union (IoU), precision, and recall.
- Threshold/post-processing comparisons and performance by nerve presence or nerve size.
- Training/validation loss curves and fold/patient variability.
- Patient-level bootstrap confidence intervals and paired comparisons for ablations.

The balanced score is intended to avoid letting the large number of empty frames dominate the assessment. Consult the generated reports for the exact definitions used by each metric.

### Explainability

The explainability step evaluates four methods:

- **Grad-CAM:** gradients of the presence-head output with respect to the deepest feature map.
- **Head-CAM:** location-wise contributions to the linear presence head.
- **Seg-Grad-CAM:** gradients of segmentation logits with respect to the final decoder features.
- **Occlusion:** measures the change in the presence score when image patches are hidden.

It also evaluates explanation maps with measures such as the pointing game, nerve-region energy/enrichment, deletion tests, weight-randomisation sanity checks, and edge reliance. These checks assess aspects of explanation behaviour; they do not establish clinical validity or prove that a model is reasoning correctly.

## Requirements

Use Python with the following packages:

```bash
pip install mlflow torch scikit-learn scipy pandas matplotlib pillow
