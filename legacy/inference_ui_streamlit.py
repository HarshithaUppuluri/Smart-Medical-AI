"""Streamlit app tying together the live-demo project components:

  - Describe Symptoms    -> NLP (TF-IDF + LogisticRegression on free text)
  - Skin Image Check     -> Deep Learning (transfer-learned ResNet18)
  - Feedback             -> Sentiment Analysis (TF-IDF + LogisticRegression)

Classification (RandomForest) and Explainable AI (SHAP) are still trained and
evaluated by model_training_testing.py / explainability.py -- see reports_assets/
for the SHAP plots and classification report used in the write-up.

Run with:  streamlit run inference_ui.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import RAW_DIR, MODELS_DIR, CONFIDENCE_THRESHOLDS, MIN_IMAGE_STD

st.set_page_config(page_title="Smart Disease Prediction & Patient Support", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------- cached loaders
@st.cache_resource
def load_nlp():
    return joblib.load(MODELS_DIR / "nlp_model.joblib")


@st.cache_resource
def load_sentiment():
    return joblib.load(MODELS_DIR / "sentiment_model.joblib")


@st.cache_resource
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


@st.cache_data
def load_disease_lookup():
    desc = pd.read_csv(RAW_DIR / "classification" / "symptom_Description.csv")
    prec = pd.read_csv(RAW_DIR / "classification" / "symptom_precaution.csv")
    desc["key"] = desc["Disease"].str.strip().str.lower()
    prec["key"] = prec["Disease"].str.strip().str.lower()
    return desc.set_index("key"), prec.set_index("key")


def show_disease_info(disease_name: str):
    desc_df, prec_df = load_disease_lookup()
    key = disease_name.strip().lower()
    if key in desc_df.index:
        st.info(desc_df.loc[key, "Description"])
    if key in prec_df.index:
        precautions = [p for p in prec_df.loc[key, ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]] if isinstance(p, str) and p.strip()]
        if precautions:
            st.markdown("**Suggested precautions:**")
            for p in precautions:
                st.markdown(f"- {p.capitalize()}")


st.title("🩺 Smart Disease Prediction and Patient Support System")
st.caption(
    "Educational prototype built for the Programming for AI module. "
    "Not a medical device — predictions are for demonstration purposes only."
)

with st.sidebar:
    st.header("About this app")
    st.markdown(
        "- **NLP**: TF-IDF + LogisticRegression on free-text symptom descriptions\n"
        "- **Deep Learning**: transfer-learned ResNet18 on skin-condition images\n"
        "- **Sentiment Analysis**: TF-IDF + LogisticRegression on patient feedback\n"
    )
    st.markdown(
        "Classification (RandomForest) and Explainable AI (SHAP) are trained/evaluated "
        "separately — see `model_training_testing.py` and `explainability.py`, with "
        "output plots in `reports_assets/`."
    )
    st.markdown("---")
    st.markdown(
        "**Data sources (Kaggle):**\n"
        "- Disease Symptom Prediction\n"
        "- Symptom2Disease\n"
        "- Skin Disease Classification Image Dataset\n"
        "- UCI ML Drug Review Dataset"
    )

tab2, tab3, tab4 = st.tabs([
    "💬 Describe Symptoms (NLP)",
    "🖼️ Skin Image Check (Deep Learning)",
    "💊 Feedback Sentiment",
])

# ---------------------------------------------------------------- Tab 2: NLP
with tab2:
    st.subheader("Describe your symptoms in your own words")
    text = st.text_area(
        "e.g. \"I've had a burning feeling when I pee and lower back pain for two days\"",
        height=120,
    )
    if st.button("Analyse my description"):
        if not text.strip():
            st.warning("Please enter a description.")
        else:
            nlp_model = load_nlp()
            proba = nlp_model.predict_proba([text.lower()])[0]
            classes = nlp_model.classes_
            top3_idx = np.argsort(-proba)[:3]
            pred = classes[top3_idx[0]]
            confidence = proba[top3_idx[0]]

            if confidence < CONFIDENCE_THRESHOLDS["nlp"]:
                st.error(
                    "⚠️ **Unable to confidently identify a condition from this description. "
                    "Please contact a doctor.**"
                )
                with st.expander("Show raw model output (low confidence — for reference only)"):
                    for i in top3_idx:
                        st.write(f"- {classes[i]} — {proba[i]*100:.1f}%")
            else:
                st.success(f"**Most likely: {pred}** ({confidence*100:.1f}% confidence)")
                chart_df = pd.DataFrame({"disease": [classes[i] for i in top3_idx], "confidence": [proba[i] for i in top3_idx]})
                st.bar_chart(chart_df.set_index("disease"))
                show_disease_info(pred)

# ---------------------------------------------------------------- Tab 3: image DL
with tab3:
    st.subheader("Upload a photo of a visible skin condition")
    st.caption("Preliminary recognition only — always consult a dermatologist for an actual diagnosis.")
    st.caption(
        "⚠️ Known limitation: this demo has not been hardened against out-of-distribution "
        "images. Blank/solid-colour uploads are rejected, but an unrelated photo can still "
        "receive a confident-looking (and wrong) prediction — a known weakness of standard "
        "softmax CNNs, discussed further in the report."
    )
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        import torch
        from torchvision import transforms

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=300)

        pixel_std = np.array(image.resize((160, 160))).std()
        if pixel_std < MIN_IMAGE_STD:
            st.error(
                "⚠️ **This image appears blank or invalid. Please upload a clear photo of "
                "the affected skin area, or contact a doctor.**"
            )
        else:
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

            top3_idx = np.argsort(-proba)[:3]
            confidence = proba[top3_idx[0]]

            if confidence < CONFIDENCE_THRESHOLDS["images"]:
                st.error(
                    "⚠️ **Unable to confidently recognise a condition from this image. "
                    "Please contact a doctor.**"
                )
                with st.expander("Show raw model output (low confidence — for reference only)"):
                    for i in top3_idx:
                        st.write(f"- {class_names[i]} — {proba[i]*100:.1f}%")
            else:
                st.success(f"**Most likely: {class_names[top3_idx[0]]}** ({confidence*100:.1f}% confidence)")
                chart_df = pd.DataFrame(
                    {"condition": [class_names[i] for i in top3_idx], "confidence": [proba[i] for i in top3_idx]}
                )
                st.bar_chart(chart_df.set_index("condition"))

# ---------------------------------------------------------------- Tab 4: sentiment
with tab4:
    st.subheader("Patient feedback sentiment")
    feedback = st.text_area("Paste a piece of patient feedback or a drug review:", height=120)
    if st.button("Analyse sentiment"):
        if not feedback.strip():
            st.warning("Please enter some feedback text.")
        else:
            sent_model = load_sentiment()
            proba = sent_model.predict_proba([feedback])[0]
            classes = sent_model.classes_
            pred = classes[np.argmax(proba)]
            confidence = proba.max()

            if confidence < CONFIDENCE_THRESHOLDS["sentiment"]:
                st.error("⚠️ **Unable to confidently classify the sentiment of this feedback.**")
                with st.expander("Show raw model output (low confidence — for reference only)"):
                    for c, p in zip(classes, proba):
                        st.write(f"- {c} — {p*100:.1f}%")
            else:
                colour = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}.get(pred, "")
                st.success(f"**Sentiment: {colour} {pred}** ({confidence*100:.1f}% confidence)")
                chart_df = pd.DataFrame({"sentiment": classes, "probability": proba})
                st.bar_chart(chart_df.set_index("sentiment"))
