import pandas as pd
import numpy as np
from mldriguard import DriftGuard

# Create synthetic data
reference_data = pd.DataFrame({
    'num_col': np.random.normal(0, 1, 100),
    'cat_col': np.random.choice(['A', 'B'], 100)
})
new_data = pd.DataFrame({
    'num_col': np.random.normal(1, 1, 100),
    'cat_col': np.random.choice(['A', 'B'], 100)
})

# Initialize and monitor
dg = DriftGuard(reference_data)
results = dg.monitor(new_data)
print(results)

# Visualize drift
dg.plot_drift()

# Suggest mitigation
print(dg.suggest_mitigation())