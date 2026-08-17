"""SHAP explainability for the disease-classification model (Task 4 requirement).

Produces:
  reports_assets/shap_summary.png            -- global: which symptoms matter most overall
  reports_assets/shap_waterfall_example.png  -- local: why one specific patient got their prediction

Also exposes `explain_prediction()` which inference_ui.py calls to show a
per-patient explanation live in the app.
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, MODELS_DIR, ASSETS_DIR, RANDOM_STATE


def load_artifacts():
    bundle = joblib.load(MODELS_DIR / "classification_model.joblib")
    model, symptom_columns, classes = bundle["model"], bundle["symptom_columns"], bundle["classes"]
    df = pd.read_csv(PROCESSED_DIR / "classification.csv")
    X = df[symptom_columns]
    return model, symptom_columns, classes, X


def explain_prediction(symptom_vector: pd.DataFrame, top_n: int = 5):
    """symptom_vector: 1-row DataFrame with the same columns as symptom_columns.

    Returns (predicted_disease, [(symptom, shap_value), ...]) for the top_n
    symptoms that pushed the model most strongly toward its prediction.
    """
    model, symptom_columns, classes, _ = load_artifacts()
    explainer = shap.TreeExplainer(model)
    sv = explainer(symptom_vector)  # Explanation: values shape (1, n_features, n_classes)

    pred = model.predict(symptom_vector)[0]
    class_idx = list(model.classes_).index(pred)

    values = sv.values[0, :, class_idx]
    order = np.argsort(-np.abs(values))[:top_n]
    contributions = [(symptom_columns[i], float(values[i])) for i in order]
    return pred, contributions


def main():
    model, symptom_columns, classes, X = load_artifacts()
    print(f"Loaded classification model: {len(symptom_columns)} symptoms, {len(classes)} diseases")

    explainer = shap.TreeExplainer(model)
    sample = X.sample(n=min(150, len(X)), random_state=RANDOM_STATE)
    sv = explainer(sample)
    print(f"SHAP values shape: {sv.values.shape}")

    # ---- global summary: mean |SHAP| per symptom, averaged across all disease classes
    mean_abs_per_class = np.abs(sv.values).mean(axis=0)   # (n_features, n_classes)
    global_importance = mean_abs_per_class.mean(axis=1)   # (n_features,)
    top_idx = np.argsort(-global_importance)[:15]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [symptom_columns[i] for i in top_idx][::-1],
        global_importance[top_idx][::-1],
        color="#2a9d8f",
    )
    plt.xlabel("mean |SHAP value| (avg across all 41 diseases)")
    plt.title("Which symptoms most influence the disease predictor overall?")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_summary.png", dpi=120)
    plt.close()
    print(f"  saved {ASSETS_DIR / 'shap_summary.png'}")

    # ---- local explanation: one example patient
    example = X.sample(n=1, random_state=RANDOM_STATE)
    pred, contributions = explain_prediction(example)
    print(f"\nExample patient -> predicted disease: {pred}")
    for sym, val in contributions:
        direction = "pushes toward" if val > 0 else "pushes away from"
        print(f"  {sym:30s} {direction} '{pred}'  (shap={val:+.3f})")

    class_idx = list(model.classes_).index(pred)
    example_sv = explainer(example)
    exp = example_sv[0, :, class_idx]

    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(f"Why this patient was predicted as: {pred}")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_waterfall_example.png", dpi=120)
    plt.close()
    print(f"  saved {ASSETS_DIR / 'shap_waterfall_example.png'}")


if __name__ == "__main__":
    main()
