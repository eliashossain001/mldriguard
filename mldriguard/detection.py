import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from .visualization import plot_drift
from .mitigation import suggest_mitigation

class DriftGuard:
    def __init__(self, reference_data, target_column=None):
        """Initialize with reference (training) data."""
        self.reference_data = reference_data.copy()
        self.target_column = target_column
        self.drift_results = {}

    def monitor(self, new_data):
        """Monitor new data for drift."""
        self.drift_results = {}
        new_data = new_data.copy()

        for column in self.reference_data.columns:
            if column == self.target_column:
                continue
            if np.issubdtype(self.reference_data[column].dtype, np.number):
                # Numerical feature: Use Kolmogorov-Smirnov test
                stat, p_value = ks_2samp(
                    self.reference_data[column].dropna(),
                    new_data[column].dropna()
                )
                self.drift_results[column] = {
                    "type": "numerical",
                    "statistic": stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05
                }
            else:
                # Categorical feature: Use Chi-squared test
                ref_counts = self.reference_data[column].value_counts()
                new_counts = new_data[column].value_counts()
                categories = list(set(ref_counts.index).union(set(new_counts.index)))
                contingency_table = [
                    [ref_counts.get(cat, 0) for cat in categories],
                    [new_counts.get(cat, 0) for cat in categories]
                ]
                stat, p_value, _, _ = chi2_contingency(contingency_table)
                self.drift_results[column] = {
                    "type": "categorical",
                    "statistic": stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05
                }
        return self.drift_results

    def plot_drift(self):
        """Visualize drift results."""
        if not self.drift_results:
            raise ValueError("Run monitor() before plotting.")
        plot_drift(self.reference_data, self.drift_results)

    def suggest_mitigation(self):
        """Suggest mitigation strategies."""
        if not self.drift_results:
            raise ValueError("Run monitor() before suggesting mitigation.")
        return suggest_mitigation(self.drift_results)