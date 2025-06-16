import pytest
import pandas as pd
import numpy as np
from mldriguard import DriftGuard

def test_drift_detection():
    # Create synthetic data
    reference_data = pd.DataFrame({
        "num_col": np.random.normal(0, 1, 100),
        "cat_col": np.random.choice(["A", "B"], 100)
    })
    new_data = pd.DataFrame({
        "num_col": np.random.normal(1, 1, 100),  # Shifted mean
        "cat_col": np.random.choice(["A", "B"], 100)
    })

    dg = DriftGuard(reference_data)
    results = dg.monitor(new_data)
    
    assert "num_col" in results
    assert "cat_col" in results
    assert results["num_col"]["drift_detected"]  # Expect Facet: Should detect drift
    assert "p_value" in results["num_col"]