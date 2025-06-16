# MLDriftGuard

![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![PyPI](https://img.shields.io/pypi/v/mldriguard)

**MLDriftGuard** is a lightweight Python package for detecting and mitigating data drift in machine learning models. Data drift occurs when the statistical properties of the input data change over time, potentially degrading model performance. This package provides tools to monitor drift, visualize distributions, and suggest mitigation strategies, making it ideal for data scientists and machine learning engineers working on deployed models.

## Features
- Detects drift in both numerical and categorical features using statistical tests (Kolmogorov-Smirnov for numerical, Chi-squared for categorical).
- Visualizes drift with kernel density estimation (KDE) plots for numerical data and bar plots for categorical data.
- Suggests mitigation strategies (e.g., retraining) when drift is detected.
- Easy to integrate into existing ML pipelines with minimal dependencies.

## Installation

Install `mldriguard` directly from PyPI:

```bash
pip install mldriguard

## Prerequisites
- Python 3.6 or higher.
- Ensure you have pip and an internet connection to install dependencies.
## Usage
Example 1: Synthetic Data (Demo)
This example replicates the test_drift.py script used during development, demonstrating drift detection with synthetic data.

```bash
import pandas as pd
import numpy as np
from mldriguard import DriftGuard

# Create synthetic reference and new data
reference_data = pd.DataFrame({
    'num_col': np.random.normal(0, 1, 100),
    'cat_col': np.random.choice(['A', 'B'], 100)
})
new_data = pd.DataFrame({
    'num_col': np.random.normal(1, 1, 100),
    'cat_col': np.random.choice(['A', 'B'], 100)
})

# Initialize DriftGuard and monitor drift
dg = DriftGuard(reference_data)
results = dg.monitor(new_data)
print("Drift Results:", results)

# Visualize drift
dg.plot_drift(new_data)

# Suggest mitigation
print("Mitigation Suggestions:", dg.suggest_mitigation())

