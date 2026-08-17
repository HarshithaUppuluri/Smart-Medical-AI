"""Flask backend for the Smart Disease Prediction & Patient Support System.

Serves a plain HTML/CSS/JS frontend (templates/ + static/) and exposes JSON
endpoints that run the trained models:

  POST /api/symptoms   -> NLP (TF-IDF + LogisticRegression) on free-text symptoms
  POST /api/image      -> Deep Learning (transfer-learned ResNet18) on a skin photo
  POST /api/sentiment  -> Sentiment Analysis (TF-IDF + LogisticRegression) on feedback
  POST /api/ocr        -> Local Tesseract OCR on a document photo

Classification (RandomForest) and Explainable AI (SHAP) are still trained and
evaluated by model_training_testing.py / explainability.py -- see reports_assets/
for the SHAP plots and classification report used in the write-up.

Run with:  .venv311/bin/python app.py
Then open: http://127.0.0.1:8000
"""
import csv
import io
import json
import shutil
import subprocess
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from config import RAW_DIR, MODELS_DIR, CONFIDENCE_THRESHOLDS, MIN_IMAGE_STD

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap
Image.MAX_IMAGE_PIXELS = 24_000_000


# ---------------------------------------------------------------- cached loaders
@lru_cache(maxsize=1)
def load_nlp():
    return joblib.load(MODELS_DIR / "nlp_model.joblib")


@lru_cache(maxsize=1)
def load_sentiment():
    return joblib.load(MODELS_DIR / "sentiment_model.joblib")


@lru_cache(maxsize=1)
def load_image_model():
    import torch
    import torch.nn as nn
    from torchvision import models

    meta = json.loads((MODELS_DIR / "image_classes.json").read_text())
    class_names = meta["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODELS_DIR / "image_model.pt", map_location="cpu"))
    model.eval()
    return model, class_names


@lru_cache(maxsize=1)
def load_disease_lookup():
    desc = pd.read_csv(RAW_DIR / "classification" / "symptom_Description.csv")
    prec = pd.read_csv(RAW_DIR / "classification" / "symptom_precaution.csv")
    desc["key"] = desc["Disease"].str.strip().str.lower()
    prec["key"] = prec["Disease"].str.strip().str.lower()
    return desc.set_index("key"), prec.set_index("key")


def disease_info(disease_name: str):
    desc_df, prec_df = load_disease_lookup()
    key = disease_name.strip().lower()
    info = {"description": None, "precautions": []}
    if key in desc_df.index:
        value = desc_df.loc[key, "Description"]
        info["description"] = value if isinstance(value, str) else None
    if key in prec_df.index:
        info["precautions"] = [
            p.strip().capitalize()
            for p in prec_df.loc[key, ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
            if isinstance(p, str) and p.strip()
        ]
    return info


def top_k(classes, proba, k=3):
    idx = np.argsort(-proba)[: min(k, len(classes))]
    return [{"label": str(classes[i]), "prob": float(proba[i])} for i in idx]


def extract_text_from_image(image: Image.Image, mode: str = "document"):
    """Run local Tesseract OCR and return readable text plus quality metadata."""
    if not shutil.which("tesseract"):
        raise RuntimeError(
            "OCR is not available on this computer. Install the Tesseract OCR "
            "engine, then restart the app."
        )

    # A larger, high-contrast grayscale image is substantially easier to read,
    # especially for phone photos of paper documents.
    from PIL import ImageEnhance, ImageFilter, ImageOps

    prepared = ImageOps.exif_transpose(image).convert("L")
    if prepared.width < 1600:
        scale = 1600 / prepared.width
        prepared = prepared.resize(
            (1600, max(1, int(prepared.height * scale))), Image.Resampling.LANCZOS
        )
    prepared = ImageOps.autocontrast(prepared, cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.25)
    prepared = prepared.filter(ImageFilter.SHARPEN)

    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG", optimize=True)
    psm = "11" if mode == "sparse" else "6"
    command = [
        "tesseract", "stdin", "stdout", "-l", "eng", "--psm", psm,
        "-c", "preserve_interword_spaces=1", "tsv",
    ]
    try:
        completed = subprocess.run(
            command,
            input=buffer.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=25,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("OCR took too long. Try a smaller or clearer image.") from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "The OCR engine could not read this image.")

    rows = csv.DictReader(
        completed.stdout.decode("utf-8", errors="replace").splitlines(),
        delimiter="\t",
    )
    lines = {}
    confidences = []
    for row in rows:
        word = (row.get("text") or "").strip()
        if not word:
            continue
        # Page edges and ruled lines are sometimes recognised as standalone
        # punctuation; omitting them keeps the editable transcript clean.
        if not any(character.isalnum() for character in word) and word not in {"—", "-"}:
            continue
        try:
            confidence = float(row.get("conf", -1))
        except (TypeError, ValueError):
            confidence = -1
        if confidence >= 0:
            confidences.append(confidence)
        key = (row.get("block_num"), row.get("par_num"), row.get("line_num"))
        lines.setdefault(key, []).append(word)

    text = "\n".join(" ".join(words) for words in lines.values()).strip()
    confidence = round(float(np.mean(confidences)), 1) if confidences else 0.0
    return text, confidence, len(confidences)


# --------------------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------------------------------- api: NLP
@app.route("/api/symptoms", methods=["POST"])
def api_symptoms():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Please enter a description."}), 400

    model = load_nlp()
    proba = model.predict_proba([text.lower()])[0]
    ranked = top_k(model.classes_, proba, 3)
    confidence = ranked[0]["prob"]
    confident = confidence >= CONFIDENCE_THRESHOLDS["nlp"]

    result = {"ok": True, "confident": confident, "top3": ranked}
    if confident:
        result["prediction"] = ranked[0]["label"]
        result["confidence"] = confidence
        result["info"] = disease_info(ranked[0]["label"])
    return jsonify(result)


# --------------------------------------------------------------------------- api: image
@app.route("/api/image", methods=["POST"])
def api_image():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "error": "Please choose an image."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"ok": False, "error": "That file doesn't look like a valid image."}), 400

    pixel_std = float(np.array(image.resize((160, 160))).std())
    if pixel_std < MIN_IMAGE_STD:
        return jsonify({
            "ok": True,
            "confident": False,
            "invalid_image": True,
            "message": (
                "This image appears blank or invalid. Please upload a clear photo "
                "of the affected skin area, or contact a doctor."
            ),
        })

    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model, class_names = load_image_model()
    x = tf(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        proba = torch.softmax(logits, dim=1)[0].numpy()

    ranked = top_k(class_names, proba, 3)
    confidence = ranked[0]["prob"]
    confident = confidence >= CONFIDENCE_THRESHOLDS["images"]

    result = {"ok": True, "confident": confident, "top3": ranked}
    if confident:
        result["prediction"] = ranked[0]["label"]
        result["confidence"] = confidence
    return jsonify(result)


# --------------------------------------------------------------------------- api: OCR
@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "error": "Please choose a document image."}), 400

    mode = request.form.get("mode", "document")
    if mode not in {"document", "sparse"}:
        mode = "document"

    try:
        image = Image.open(file.stream)
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify({"ok": False, "error": "Please upload a valid JPG or PNG image."}), 400

    if image.width < 120 or image.height < 120:
        return jsonify({
            "ok": False,
            "error": "That image is too small for reliable OCR. Use an image at least 120 × 120 pixels.",
        }), 400

    try:
        text, confidence, word_count = extract_text_from_image(image, mode)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503

    if not text:
        return jsonify({
            "ok": True,
            "text": "",
            "confidence": 0,
            "word_count": 0,
            "message": "No readable text was found. Try a sharper, well-lit image taken straight-on.",
        })

    return jsonify({
        "ok": True,
        "text": text,
        "confidence": confidence,
        "word_count": word_count,
        "character_count": len(text),
    })


# --------------------------------------------------------------------------- api: sentiment
@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Please enter some feedback text."}), 400

    model = load_sentiment()
    proba = model.predict_proba([text])[0]
    ranked = top_k(model.classes_, proba, len(model.classes_))
    confidence = ranked[0]["prob"]
    confident = confidence >= CONFIDENCE_THRESHOLDS["sentiment"]

    result = {"ok": True, "confident": confident, "top3": ranked}
    if confident:
        result["prediction"] = ranked[0]["label"]
        result["confidence"] = confidence
    return jsonify(result)


if __name__ == "__main__":
    # Port 5000 is claimed by macOS AirPlay Receiver (Control Center) on modern
    # macOS, which answers non-AirPlay requests with "access denied" -- 8000
    # avoids that collision.
    app.run(debug=True, port=8000)
