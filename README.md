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
