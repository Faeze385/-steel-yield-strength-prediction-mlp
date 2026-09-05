# Steel Yield Strength Prediction via Deep Learning

A reproducible PyTorch machine learning pipeline that predicts steel alloy yield strength using composition-based features generated through `matminer` and `pymatgen`.

## Project Overview
Predicting mechanical properties from material chemical compositions bridges physical metallurgy and artificial intelligence. This repository utilizes the `matbench_steels` benchmark dataset, extracts standardized attribute descriptors (`Magpie`), and implements a custom Multi-Layer Perceptron (MLP) in PyTorch to estimate alloy yield strength.

## Key Features & Methodology
* **Featurization**: Extracts physical and chemical descriptors from alloy chemical formulas using `matminer`'s Magpie element property preset.
* **Data Preprocessing**: Normalizes input feature matrices using `MinMaxScaler` and initializes random number generators with fixed seeds (`torch.manual_seed(42)`) to ensure complete experimental reproducibility.
* **Deep Learning Architecture**: Utilizes a custom feed-forward neural network (`SteelMLP`) structured with hidden layers and non-linear `ReLU` activation functions optimized via Adam.

## Repository Structure
```text
├── steel-yield-strength-prediction.ipynb  # Complete Jupyter notebook containing data pipeline and model
└── README.md                              # Project documentation
