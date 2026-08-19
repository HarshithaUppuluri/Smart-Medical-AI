"""
Smart Medical AI
Flask Backend for the Smart Disease Prediction & Patient Support System.

This file connects the frontend web application with the trained AI and
Machine Learning models.

The application provides the following AI features:

1. Natural Language Processing (NLP)
   POST /api/symptoms
   Uses TF-IDF and Logistic Regression to analyse free-text symptoms
   and predict possible diseases.

2. Deep Learning
   POST /api/image
   Uses a trained ResNet18 neural network to classify uploaded
   skin-condition images.

3. Sentiment Analysis
   POST /api/sentiment
   Uses TF-IDF and Logistic Regression to classify user feedback.

4. Optical Character Recognition (OCR)
   POST /api/ocr
   Uses Tesseract OCR to extract readable text from document images.
   OCR is implemented as the advanced AI feature of the project.

Classification using Random Forest and Explainable AI using SHAP are
trained and evaluated separately in model_training_testing.py and
explainability.py.

Run the application using:

    python app.py

Then open:

    http://127.0.0.1:8000
"""


# ===========================================================================
# STANDARD PYTHON LIBRARIES
# ===========================================================================

import csv
import io
import json
import shutil
import subprocess

from functools import lru_cache


# ===========================================================================
# THIRD-PARTY LIBRARIES
# ===========================================================================

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError


# ===========================================================================
# PROJECT CONFIGURATION
# ===========================================================================

# Import central project settings such as dataset paths, model paths and
# confidence thresholds.
from config import (
    RAW_DIR,
    MODELS_DIR,
    CONFIDENCE_THRESHOLDS,
    MIN_IMAGE_STD,
)


# ===========================================================================
# FLASK APPLICATION SETUP
# ===========================================================================

# Create the Flask web application.
app = Flask(__name__)

# Restrict uploaded files to a maximum of 8 MB.
# This reduces the risk of excessively large files consuming server resources.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Prevent Pillow from processing extremely large images.
Image.MAX_IMAGE_PIXELS = 24_000_000


# ===========================================================================
# MODEL LOADING
# ===========================================================================
#
# Models are cached after being loaded for the first time.
# This prevents the application from repeatedly loading large model files
# from disk for every request and improves response time.
# ===========================================================================


@lru_cache(maxsize=1)
def load_nlp():
    """
    Load and cache the trained NLP disease-prediction model.

    The model uses TF-IDF text features and Logistic Regression to predict
    diseases from free-text symptom descriptions.
    """

    return joblib.load(
        MODELS_DIR / "nlp_model.joblib"
    )


@lru_cache(maxsize=1)
def load_sentiment():
    """
    Load and cache the trained sentiment-analysis model.

    The model analyses text feedback and predicts its sentiment category.
    """

    return joblib.load(
        MODELS_DIR / "sentiment_model.joblib"
    )


@lru_cache(maxsize=1)
def load_image_model():
    """
    Load and cache the trained ResNet18 skin-image classification model.

    Returns:
        model:
            Trained PyTorch ResNet18 model.

        class_names:
            List containing the skin-condition classes recognised by
            the model.
    """

    import torch
    import torch.nn as nn

    from torchvision import models

    # Read the skin-condition labels that were used during model training.
    meta = json.loads(
        (MODELS_DIR / "image_classes.json").read_text()
    )

    class_names = meta["classes"]

    # Recreate the ResNet18 neural network architecture.
    model = models.resnet18(
        weights=None
    )

    # Replace the final classification layer so that the number of outputs
    # matches the number of skin-condition classes.
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names),
    )

    # Load the trained model parameters.
    # map_location="cpu" allows the model to run without requiring a GPU.
    model.load_state_dict(
        torch.load(
            MODELS_DIR / "image_model.pt",
            map_location="cpu",
        )
    )

    # Switch the network to evaluation mode because the application is
    # performing predictions rather than training.
    model.eval()

    return model, class_names


# ===========================================================================
# DISEASE INFORMATION LOOKUP
# ===========================================================================


@lru_cache(maxsize=1)
def load_disease_lookup():
    """
    Load disease descriptions and precaution information from CSV files.

    Returns:
        tuple:
            Disease-description DataFrame and disease-precaution DataFrame.
    """

    # Load disease description information.
    desc = pd.read_csv(
        RAW_DIR
        / "classification"
        / "symptom_Description.csv"
    )

    # Load disease precaution information.
    prec = pd.read_csv(
        RAW_DIR
        / "classification"
        / "symptom_precaution.csv"
    )

    # Create normalised disease-name keys.
    # Lowercase text and stripped whitespace make matching more reliable.
    desc["key"] = (
        desc["Disease"]
        .str.strip()
        .str.lower()
    )

    prec["key"] = (
        prec["Disease"]
        .str.strip()
        .str.lower()
    )

    return (
        desc.set_index("key"),
        prec.set_index("key"),
    )


def disease_info(disease_name: str):
    """
    Retrieve the description and precautions associated with a disease.

    Args:
        disease_name:
            Disease predicted by the NLP model.

    Returns:
        Dictionary containing the disease description and recommended
        precautions.
    """

    desc_df, prec_df = load_disease_lookup()

    # Normalise the predicted disease name before searching the datasets.
    key = disease_name.strip().lower()

    # Default response in case no supporting information is available.
    info = {
        "description": None,
        "precautions": [],
    }

    # Retrieve the disease description when available.
    if key in desc_df.index:

        value = desc_df.loc[
            key,
            "Description",
        ]

        info["description"] = (
            value
            if isinstance(value, str)
            else None
        )

    # Retrieve up to four recommended precautions.
    if key in prec_df.index:

        info["precautions"] = [
            p.strip().capitalize()

            for p in prec_df.loc[
                key,
                [
                    "Precaution_1",
                    "Precaution_2",
                    "Precaution_3",
                    "Precaution_4",
                ],
            ]

            if isinstance(p, str)
            and p.strip()
        ]

    return info


# ===========================================================================
# PREDICTION HELPER FUNCTION
# ===========================================================================


def top_k(classes, proba, k=3):
    """
    Return the highest-probability predictions.

    Args:
        classes:
            Model class labels.

        proba:
            Probability associated with each class.

        k:
            Number of predictions to return.

    Returns:
        List containing class labels and their probabilities.
    """

    # Sort probabilities from highest to lowest.
    idx = np.argsort(
        -proba
    )[: min(k, len(classes))]

    return [
        {
            "label": str(classes[i]),
            "prob": float(proba[i]),
        }
        for i in idx
    ]


# ===========================================================================
# ADVANCED AI FEATURE: OPTICAL CHARACTER RECOGNITION
# ===========================================================================
#
# Optical Character Recognition allows the application to extract digital
# text from uploaded images of documents.
#
# The uploaded image is first preprocessed to improve readability.
# Tesseract OCR then detects individual words and returns text together
# with confidence values.
#
# OCR is used as the advanced AI feature beyond the main topics covered
# within the module.
# ===========================================================================


def extract_text_from_image(
    image: Image.Image,
    mode: str = "document",
):
    """
    Extract text from a document image using Tesseract OCR.

    Image preprocessing includes:
    - correcting EXIF orientation,
    - grayscale conversion,
    - resizing,
    - automatic contrast adjustment,
    - contrast enhancement,
    - image sharpening.

    Args:
        image:
            Pillow image uploaded by the user.

        mode:
            OCR segmentation mode.

            "document" assumes structured text.

            "sparse" is designed for text positioned in different areas
            of the image.

    Returns:
        tuple containing:
            extracted text,
            average OCR confidence,
            recognised word count.
    """

    # Check that Tesseract OCR is installed on the computer.
    if not shutil.which("tesseract"):

        raise RuntimeError(
            "OCR is not available on this computer. "
            "Install the Tesseract OCR engine, then restart the app."
        )

    from PIL import (
        ImageEnhance,
        ImageFilter,
        ImageOps,
    )

    # Correct image orientation and convert the image to grayscale.
    prepared = (
        ImageOps.exif_transpose(image)
        .convert("L")
    )

    # Increase the resolution of small images because larger text is
    # generally easier for Tesseract to recognise.
    if prepared.width < 1600:

        scale = (
            1600
            / prepared.width
        )

        prepared = prepared.resize(
            (
                1600,
                max(
                    1,
                    int(
                        prepared.height
                        * scale
                    ),
                ),
            ),
            Image.Resampling.LANCZOS,
        )

    # Automatically improve image contrast.
    prepared = ImageOps.autocontrast(
        prepared,
        cutoff=1,
    )

    # Increase text/background contrast.
    prepared = (
        ImageEnhance
        .Contrast(prepared)
        .enhance(1.25)
    )

    # Sharpen character edges to improve OCR recognition.
    prepared = prepared.filter(
        ImageFilter.SHARPEN
    )

    # Store the processed image in memory as PNG data.
    buffer = io.BytesIO()

    prepared.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    # Tesseract page-segmentation settings:
    #
    # PSM 6:
    # Assume a single uniform block of text.
    #
    # PSM 11:
    # Identify sparse text at different locations.
    psm = (
        "11"
        if mode == "sparse"
        else "6"
    )

    # Build the Tesseract command.
    # TSV output is requested so that word-level confidence scores
    # are available.
    command = [
        "tesseract",
        "stdin",
        "stdout",
        "-l",
        "eng",
        "--psm",
        psm,
        "-c",
        "preserve_interword_spaces=1",
        "tsv",
    ]

    try:

        # Run the OCR engine and send the processed image directly
        # through standard input.
        completed = subprocess.run(
            command,
            input=buffer.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=25,
        )

    except subprocess.TimeoutExpired as exc:

        raise RuntimeError(
            "OCR took too long. "
            "Try a smaller or clearer image."
        ) from exc

    # If Tesseract fails, return its error message.
    if completed.returncode != 0:

        detail = (
            completed.stderr
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )

        raise RuntimeError(
            detail
            or "The OCR engine could not read this image."
        )

    # Parse the TSV output generated by Tesseract.
    rows = csv.DictReader(
        completed.stdout
        .decode(
            "utf-8",
            errors="replace",
        )
        .splitlines(),
        delimiter="\t",
    )

    # Store recognised text lines.
    lines = {}

    # Store valid OCR confidence values.
    confidences = []

    for row in rows:

        word = (
            row.get("text")
            or ""
        ).strip()

        # Ignore empty words.
        if not word:
            continue

        # Ignore standalone punctuation that may have been incorrectly
        # detected from document borders or lines.
        if (
            not any(
                character.isalnum()
                for character in word
            )
            and word not in {
                "—",
                "-",
            }
        ):
            continue

        # Read the confidence score associated with the detected word.
        try:

            confidence = float(
                row.get(
                    "conf",
                    -1,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = -1

        # Negative confidence indicates an invalid OCR result.
        if confidence >= 0:
            confidences.append(
                confidence
            )

        # Group words using their OCR block, paragraph and line numbers.
        key = (
            row.get("block_num"),
            row.get("par_num"),
            row.get("line_num"),
        )

        lines.setdefault(
            key,
            [],
        ).append(
            word
        )

    # Reconstruct readable text while preserving detected line breaks.
    text = "\n".join(
        " ".join(words)
        for words in lines.values()
    ).strip()

    # Calculate the average confidence across all recognised words.
    confidence = (
        round(
            float(
                np.mean(
                    confidences
                )
            ),
            1,
        )
        if confidences
        else 0.0
    )

    return (
        text,
        confidence,
        len(confidences),
    )


# ===========================================================================
# MAIN WEB PAGE
# ===========================================================================


@app.route("/")
def index():
    """
    Render the main Smart Medical AI web interface.
    """

    return render_template(
        "index.html"
    )


# ===========================================================================
# NLP SYMPTOM PREDICTION API
# ===========================================================================
#
# This API accepts a natural-language symptom description.
#
# Example:
# "I have a headache, fever and feel tired."
#
# The trained NLP pipeline converts the text into TF-IDF features and uses
# Logistic Regression to calculate disease probabilities.
# ===========================================================================


@app.route(
    "/api/symptoms",
    methods=["POST"],
)
def api_symptoms():
    """
    Analyse free-text symptoms and predict possible diseases.
    """

    # Read JSON data sent by the frontend.
    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    # Extract and clean the symptom description.
    text = (
        data.get("text")
        or ""
    ).strip()

    # Prevent prediction when the user has not entered symptoms.
    if not text:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Please enter a description."
                ),
            }
        ), 400

    # Load the trained NLP model.
    model = load_nlp()

    # Calculate the probability associated with every disease class.
    proba = model.predict_proba(
        [
            text.lower()
        ]
    )[0]

    # Select the three highest probability predictions.
    ranked = top_k(
        model.classes_,
        proba,
        3,
    )

    # Highest probability becomes the model confidence.
    confidence = ranked[0]["prob"]

    # Determine whether confidence is high enough to display
    # a final disease prediction.
    confident = (
        confidence
        >= CONFIDENCE_THRESHOLDS[
            "nlp"
        ]
    )

    # Return the ranked predictions to the frontend.
    result = {
        "ok": True,
        "confident": confident,
        "top3": ranked,
    }

    # Only return detailed disease information when confidence exceeds
    # the configured threshold.
    if confident:

        result["prediction"] = (
            ranked[0]["label"]
        )

        result["confidence"] = (
            confidence
        )

        result["info"] = disease_info(
            ranked[0]["label"]
        )

    return jsonify(
        result
    )


# ===========================================================================
# DEEP LEARNING SKIN IMAGE CLASSIFICATION API
# ===========================================================================
#
# This endpoint accepts a skin image from the user.
#
# The uploaded image is resized and normalised before being passed into
# the trained ResNet18 deep-learning model.
# ===========================================================================


@app.route(
    "/api/image",
    methods=["POST"],
)
def api_image():
    """
    Analyse an uploaded skin image using the trained ResNet18 model.
    """

    # Retrieve the uploaded image.
    file = request.files.get(
        "image"
    )

    # Ensure that an image has been selected.
    if (
        file is None
        or file.filename == ""
    ):

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Please choose an image."
                ),
            }
        ), 400

    try:

        # Load the image and convert it to RGB format.
        image = (
            Image.open(
                file.stream
            )
            .convert("RGB")
        )

    except UnidentifiedImageError:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "That file doesn't look "
                    "like a valid image."
                ),
            }
        ), 400

    # Calculate image pixel variation.
    #
    # Extremely low variation usually indicates that the image is blank
    # or unsuitable for classification.
    pixel_std = float(
        np.array(
            image.resize(
                (
                    160,
                    160,
                )
            )
        ).std()
    )

    # Reject blank or invalid images.
    if pixel_std < MIN_IMAGE_STD:

        return jsonify(
            {
                "ok": True,
                "confident": False,
                "invalid_image": True,
                "message": (
                    "This image appears blank or invalid. "
                    "Please upload a clear photo of the affected "
                    "skin area, or contact a doctor."
                ),
            }
        )

    import torch

    from torchvision import transforms

    # Create the image preprocessing pipeline used before prediction.
    tf = transforms.Compose(
        [
            # Resize the image to the dimensions expected by the model.
            transforms.Resize(
                (
                    160,
                    160,
                )
            ),

            # Convert the image into a PyTorch tensor.
            transforms.ToTensor(),

            # Apply ImageNet normalisation values used by ResNet.
            transforms.Normalize(
                [
                    0.485,
                    0.456,
                    0.406,
                ],
                [
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    # Load the trained image model and class names.
    model, class_names = (
        load_image_model()
    )

    # Convert the image to a model-ready tensor and add a batch dimension.
    x = tf(
        image
    ).unsqueeze(
        0
    )

    # Disable gradient calculations during prediction.
    with torch.no_grad():

        logits = model(
            x
        )

        # Convert raw neural network output values into probabilities.
        proba = (
            torch.softmax(
                logits,
                dim=1,
            )[0]
            .numpy()
        )

    # Retrieve the three highest-probability predictions.
    ranked = top_k(
        class_names,
        proba,
        3,
    )

    confidence = (
        ranked[0]["prob"]
    )

    # Check whether the prediction satisfies the required confidence level.
    confident = (
        confidence
        >= CONFIDENCE_THRESHOLDS[
            "images"
        ]
    )

    result = {
        "ok": True,
        "confident": confident,
        "top3": ranked,
    }

    # Return the primary prediction when the model is sufficiently confident.
    if confident:

        result["prediction"] = (
            ranked[0]["label"]
        )

        result["confidence"] = (
            confidence
        )

    return jsonify(
        result
    )


# ===========================================================================
# OCR DOCUMENT PROCESSING API
# ===========================================================================
#
# OCR is the advanced AI feature implemented within the application.
#
# Users can upload an image containing text such as a medical document.
# The system validates the image and uses Tesseract OCR to convert the
# image-based text into editable digital text.
# ===========================================================================


@app.route(
    "/api/ocr",
    methods=["POST"],
)
def api_ocr():
    """
    Process an uploaded document image and extract text using OCR.
    """

    # Retrieve the uploaded document image.
    file = request.files.get(
        "image"
    )

    # Reject requests where no document has been selected.
    if (
        file is None
        or file.filename == ""
    ):

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Please choose a document image."
                ),
            }
        ), 400

    # Read the selected OCR processing mode.
    mode = request.form.get(
        "mode",
        "document",
    )

    # Reject unsupported mode values and fall back to document mode.
    if mode not in {
        "document",
        "sparse",
    }:

        mode = "document"

    try:

        # Load the uploaded image.
        image = Image.open(
            file.stream
        )

        # Force Pillow to fully load the image so file errors are caught.
        image.load()

        # Convert the image to a consistent RGB format.
        image = image.convert(
            "RGB"
        )

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Please upload a valid JPG or PNG image."
                ),
            }
        ), 400

    # Very small images may not contain enough detail for reliable OCR.
    if (
        image.width < 120
        or image.height < 120
    ):

        return jsonify(
            {
                "ok": False,
                "error": (
                    "That image is too small for reliable OCR. "
                    "Use an image at least 120 × 120 pixels."
                ),
            }
        ), 400

    try:

        # Run OCR preprocessing and text extraction.
        (
            text,
            confidence,
            word_count,
        ) = extract_text_from_image(
            image,
            mode,
        )

    except RuntimeError as exc:

        # Return a service error when OCR cannot successfully run.
        return jsonify(
            {
                "ok": False,
                "error": str(
                    exc
                ),
            }
        ), 503

    # If OCR ran successfully but found no text, return a helpful message.
    if not text:

        return jsonify(
            {
                "ok": True,
                "text": "",
                "confidence": 0,
                "word_count": 0,
                "message": (
                    "No readable text was found. "
                    "Try a sharper, well-lit image "
                    "taken straight-on."
                ),
            }
        )

    # Return recognised text and useful OCR metadata.
    return jsonify(
        {
            "ok": True,
            "text": text,
            "confidence": confidence,
            "word_count": word_count,
            "character_count": len(
                text
            ),
        }
    )


# ===========================================================================
# SENTIMENT ANALYSIS API
# ===========================================================================
#
# This endpoint accepts written feedback from the user.
#
# The sentiment model analyses the text and predicts its sentiment category.
# Probability values are also returned so that the interface can display
# model confidence.
# ===========================================================================


@app.route(
    "/api/sentiment",
    methods=["POST"],
)
def api_sentiment():
    """
    Analyse user feedback using the trained sentiment-analysis model.
    """

    # Read the JSON request submitted by the frontend.
    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    # Extract and clean the feedback text.
    text = (
        data.get("text")
        or ""
    ).strip()

    # Prevent prediction when no feedback has been entered.
    if not text:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Please enter some feedback text."
                ),
            }
        ), 400

    # Load the trained sentiment model.
    model = load_sentiment()

    # Calculate probabilities for all sentiment classes.
    proba = model.predict_proba(
        [
            text
        ]
    )[0]

    # Rank sentiment classes according to probability.
    ranked = top_k(
        model.classes_,
        proba,
        len(
            model.classes_
        ),
    )

    # Highest probability represents model confidence.
    confidence = (
        ranked[0]["prob"]
    )

    # Apply the configured sentiment confidence threshold.
    confident = (
        confidence
        >= CONFIDENCE_THRESHOLDS[
            "sentiment"
        ]
    )

    result = {
        "ok": True,
        "confident": confident,
        "top3": ranked,
    }

    # Return the top sentiment when confidence is sufficiently high.
    if confident:

        result["prediction"] = (
            ranked[0]["label"]
        )

        result["confidence"] = (
            confidence
        )

    return jsonify(
        result
    )


# ===========================================================================
# APPLICATION ENTRY POINT
# ===========================================================================


if __name__ == "__main__":

    # Start Flask's local development server.
    #
    # Port 8000 is used instead of port 5000 because port 5000 may already
    # be occupied by the AirPlay Receiver on some macOS systems.
    #
    # Debug mode automatically reloads the server when code is changed
    # during development.
    app.run(
        debug=True,
        port=8000,
    )