# GrowliFlower Image Classification

This project demonstrates deep learning-based classification of GrowliFlower images using PyTorch. The workflow includes data loading, preprocessing, model building (custom CNN, MC Dropout, and transfer learning), training, evaluation, and conformal prediction analysis.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation & Visualization](#evaluation--visualization)
- [Conformal Prediction & OOD](#conformal-prediction--ood)
- [Testing](#testing)
- [Experiment Scripts](#experiment-scripts)
- [Configuration & Notebooks](#configuration--notebooks)
- [Usage](#usage)

## Overview
- **Notebook:** `GrowliFLower_My_Model.ipynb`
- **Goal:** Classify GrowliFlower images using deep learning (custom CNN, MC Dropout, DenseNet169 transfer learning) and analyze uncertainty and conformal prediction sets.
- **Framework:** PyTorch

## Requirements
Install the following Python packages:
- torch
- torchvision
- numpy
- pandas
- matplotlib
- pillow
- scikit-learn
- opencv-python
- seaborn
- pytest (for testing)

You can install them with:
```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn opencv-python seaborn pytest
```

## Project Structure
```
ICP/
├── data_preparation.py         # Data loading, preprocessing, and DataLoader creation
├── models.py                   # Model architectures: LeNet, LeNet_MC (MC Dropout), transfer learning
├── train.py                    # Training loops, early stopping, MC Dropout training
├── eval.py                     # Evaluation utilities, MC Dropout sampling, metrics
├── visualization.py            # General training/testing/MC Dropout plots
├── ood.py                      # OOD/uncertainty analysis and visualizations
├── conformalization/
│   ├── icp_data_processing.py  # Data splitting and shape checks for conformal prediction
│   ├── conformalize.py         # Conformal prediction set construction and coverage
│   ├── icp_eval.py             # Repeated empirical coverage evaluation
│   ├── icp_inference.py        # Inference utilities, merging val/test, set type analysis
│   ├── icp_visualize.py        # Conformal prediction set and ablation visualizations
│   └── icp_ablation.py         # Alpha ablation study for conformal prediction
├── tests/                      # Unit tests for utilities and modules
├── scripts/                    # Example and experiment scripts
├── notebooks/                  # Exploratory and research notebooks
└── main.py                     # Entrypoint script to run the workflow
```

## Data Preparation
- Place your GrowliFlower image data and label files in the specified directory (update `main_dr` in the code as needed).
- The code loads and concatenates label files, then splits data into training, validation, and test sets.
- Images are resized to 256x256 and normalized.

## Model Architectures
- **Custom CNN (LeNet-like):**
  - Three convolutional blocks with batch normalization and max pooling.
  - Fully connected layer for classification.
- **MC Dropout (LeNet_MC):**
  - Same as LeNet, but with Dropout2d for uncertainty estimation.
- **Transfer Learning:**
  - DenseNet169 with frozen base layers.
  - Custom classifier head for binary classification.

## Training
- Loss: CrossEntropyLoss (with class balancing)
- Optimizer: Adam
- Early stopping based on validation loss
- Training and validation loss/accuracy are tracked and plotted.
- MC Dropout training and sampling supported.

## Evaluation & Visualization
- Plot training/validation loss curves, accuracy vs. threshold, MC Dropout accuracy distributions, and more.
- OOD analysis and uncertainty visualization utilities.
- Conformal prediction set visualizations and ablation studies.

## Conformal Prediction & OOD
- Modular code for splitting calibration/test sets, constructing conformal prediction sets, and evaluating empirical coverage.
- Alpha ablation studies and sensitivity analysis.
- OOD analysis with black box occlusion and MC Dropout uncertainty.

## Testing
- The `tests/` folder contains unit tests for utilities and modules (see `test_utils.py` for an example).
- Run all tests with:
```bash
pytest tests/
```

## Experiment Scripts
- The `scripts/` folder contains example and experiment scripts (see `example_experiment.py`).
- Use these scripts as templates for running new experiments or pipelines.

## Configuration & Notebooks
- For large projects, consider using a config file (YAML/JSON) or a config module for hyperparameters and paths.
- The `notebooks/` folder is for exploratory and research notebooks.
- All modules and functions include docstrings; for larger projects, consider generating documentation with Sphinx or MkDocs.

## Usage
1. Update the data directory path in the code as needed.
2. Use `main.py` or run individual modules for:
   - Data preparation
   - Model training (standard, MC Dropout, or transfer learning)
   - Evaluation and visualization
   - Conformal prediction and OOD analysis
3. Adjust hyperparameters (batch size, epochs, learning rate, alpha, etc.) as desired.

---

For more details, see the code and comments in each module and the `GrowliFLower_My_Model.ipynb` notebook.
