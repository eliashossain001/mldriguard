def suggest_mitigation(drift_results):
    """Suggest mitigation strategies based on drift results."""
    suggestions = []
    drifted_features = [col for col, res in drift_results.items() if res["drift_detected"]]
    
    if not drifted_features:
        return "No drift detected. Model performance is likely stable."
    
    suggestions.append(f"Drift detected in features: {', '.join(drifted_features)}.")
    suggestions.append("Consider retraining the model with recent data.")
    suggestions.append("Alternatively, apply domain adaptation techniques.")
    return "\n".join(suggestions)