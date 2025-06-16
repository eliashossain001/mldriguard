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
```
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

```

## Expected Output:

Results will contain a dictionary with p_value and drift_detected for each column.
A plot will show KDE for num_col and bars for cat_col, with "Drift: True" if detected.
Mitigation suggestions (e.g., "Consider retraining") if drift is present.

![image](https://github.com/user-attachments/assets/a48efba5-6afa-42b0-a2de-895bb6621be0)


## Example 2: Real-World Data
This example shows how to use mldriguard with a CSV file containing real data (e.g., sales figures and regions).

```bash
import pandas as pd
from mldriguard import DriftGuard

# Load real-world data
reference_data = pd.read_csv("reference_sales.csv")  # Columns: 'sales' (numerical), 'region' (categorical)
new_data = pd.read_csv("new_sales.csv")  # Same structure

# Initialize and monitor
dg = DriftGuard(reference_data)
results = dg.monitor(new_data)
print("Drift Results:", results)

# Visualize
dg.plot_drift(new_data)

# Suggest mitigation
print("Mitigation Suggestions:", dg.suggest_mitigation())

```

## Notes:

Replace reference_sales.csv and new_sales.csv with your data files.
Ensure column names match between datasets.
Customizing Drift Detection
Significance Level: By default, drift is detected at a p-value threshold of 0.05. You can adjust this by modifying the source code or subclassing DriftGuard (future updates may include a parameter).
Data Types: The package automatically handles numerical and categorical columns. Ensure your data is clean (no missing values may cause errors unless handled).

```bash
Dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
These are installed automatically with the package.

```

## Contributing
We welcome contributions to improve MLDriftGuard!


## Author
Elias Hossain. 
Email: elias.hosssain191@gmail.com

