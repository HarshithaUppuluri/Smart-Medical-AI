"""Shared paths and constants used across all pipeline scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "reports_assets"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Kaggle dataset refs -> local subfolder under data/raw
DATASETS = {
    "classification": "itachi9604/disease-symptom-description-dataset",
    "nlp": "niyarrbarman/symptom2disease",
    "images": "riyaelizashaju/skin-disease-classification-image-dataset",
    "sentiment": "jessicali9530/kuc-hackathon-winter-2018",
}

RANDOM_STATE = 42

# below these max-probability thresholds, the UI shows "unable to confidently
# predict, please consult a doctor" instead of a specific diagnosis. Values were
# picked by testing genuinely ambiguous/out-of-domain inputs against each model:
# real/confident predictions scored 51-97%, nonsense inputs scored 12-55%.
CONFIDENCE_THRESHOLDS = {
    "classification": 0.45,
    "nlp": 0.45,
    "sentiment": 0.60,
    "images": 0.60,
}

# images with pixel std below this are almost certainly blank/solid-colour, not
# a real photo -- catches the most obvious invalid uploads. NOTE: this does NOT
# catch all out-of-distribution images (e.g. random noise has high std and still
# fools the model with >95% confidence) -- see report for discussion.
MIN_IMAGE_STD = 5.0
