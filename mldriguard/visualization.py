import matplotlib.pyplot as plt
import seaborn as sns

def plot_drift(reference_data, drift_results):
    """Plot drift results for each feature."""
    plt.figure(figsize=(12, 6))
    for i, (column, result) in enumerate(drift_results.items()):
        plt.subplot(1, len(drift_results), i + 1)
        if result["type"] == "numerical":
            sns.kdeplot(reference_data[column], label="Reference")
            plt.title(f"{column}\nDrift: {result['drift_detected']}")
            plt.xlabel(column)
            plt.legend()
        else:
            counts = reference_data[column].value_counts()
            counts.plot(kind="bar", alpha=0.5, label="Reference")
            plt.title(f"{column}\nDrift: {result['drift_detected']}")
            plt.xlabel(column)
            plt.legend()
    plt.tight_layout()
    plt.show()