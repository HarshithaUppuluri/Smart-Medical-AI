"""Trains + hyperparameter-tunes all 4 models and saves them to models/.

  1. Classification   : RandomForest on one-hot symptoms -> disease   (GridSearchCV)
  2. NLP               : TF-IDF + LogisticRegression on free-text symptoms -> disease (GridSearchCV)
  3. Sentiment         : TF-IDF + LogisticRegression on drug reviews -> sentiment (GridSearchCV)
  4. Deep Learning     : transfer-learned ResNet18 on skin-condition images (PyTorch)

Each section prints its tuned hyperparameters, test-set accuracy/F1, and a full
classification report so results can be dropped straight into the report.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from config import PROCESSED_DIR, RAW_DIR, MODELS_DIR, ASSETS_DIR, RANDOM_STATE

# minimum test-set accuracy we require before accepting a model
ACCEPTED_ACCURACY = {
    "classification": 0.85,
    "nlp": 0.80,
    "sentiment": 0.55,
    "images": 0.55,
}


def save_report(name, report: str):
    (ASSETS_DIR / f"{name}_classification_report.txt").write_text(report)


# ---------------------------------------------------------------- 1. classification
def train_classification():
    print("\n[1/4] Classification: RandomForest on symptoms -> disease")
    df = pd.read_csv(PROCESSED_DIR / "classification.csv")

    # the source data repeats each unique symptom-combo ~16x on average (4920 rows,
    # only 301 unique combos). splitting on the raw rows lets identical combos leak
    # into both train and test, which is memorization, not generalization -- so we
    # dedupe to unique (symptom-set, disease) pairs *before* splitting.
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=["Disease"]), df["Disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [None, 15],
            "min_samples_split": [2, 4],
        },
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  best params: {grid.best_params_}")
    print(f"  test accuracy={acc:.4f}  macro-F1={f1:.4f}")
    assert acc >= ACCEPTED_ACCURACY["classification"], "classification model below accepted accuracy"

    save_report("classification", classification_report(y_test, y_pred))
    joblib.dump(
        {"model": best, "symptom_columns": list(X.columns), "classes": sorted(y.unique())},
        MODELS_DIR / "classification_model.joblib",
    )
    print("  saved models/classification_model.joblib")
    return acc


# ---------------------------------------------------------------- 2. nlp
def train_nlp():
    print("\n[2/4] NLP: TF-IDF + LogisticRegression on free-text symptoms -> disease")
    df = pd.read_csv(PROCESSED_DIR / "nlp.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "clf__C": [1, 10, 50],
        },
        cv=5,
        n_jobs=-1,
        scoring="f1_macro",
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  best params: {grid.best_params_}")
    print(f"  test accuracy={acc:.4f}  macro-F1={f1:.4f}")
    assert acc >= ACCEPTED_ACCURACY["nlp"], "nlp model below accepted accuracy"

    save_report("nlp", classification_report(y_test, y_pred))
    joblib.dump(best, MODELS_DIR / "nlp_model.joblib")
    print("  saved models/nlp_model.joblib")
    return acc


# ---------------------------------------------------------------- 3. sentiment
def train_sentiment():
    print("\n[3/4] Sentiment: TF-IDF + LogisticRegression on drug reviews")
    df = pd.read_csv(PROCESSED_DIR / "sentiment.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__C": [1, 5, 10],
        },
        cv=3,
        n_jobs=-1,
        scoring="f1_macro",
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  best params: {grid.best_params_}")
    print(f"  test accuracy={acc:.4f}  macro-F1={f1:.4f}")
    assert acc >= ACCEPTED_ACCURACY["sentiment"], "sentiment model below accepted accuracy"

    save_report("sentiment", classification_report(y_test, y_pred))
    joblib.dump(best, MODELS_DIR / "sentiment_model.joblib")
    print("  saved models/sentiment_model.joblib")
    return acc


# ---------------------------------------------------------------- 4. deep learning (images)
def train_images():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    print("\n[4/4] Deep Learning: transfer-learned ResNet18 on skin condition images")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device: {device}")

    data_dir = RAW_DIR / "images" / "Split_smol"
    train_tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    class_names = train_ds.classes
    print(f"  {len(train_ds)} train / {len(val_ds)} val images, {len(class_names)} classes")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    def build_model(unfreeze_layer4: bool):
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in m.parameters():
            param.requires_grad = False
        if unfreeze_layer4:
            for param in m.layer4.parameters():
                param.requires_grad = True
        m.fc = nn.Linear(m.fc.in_features, len(class_names))
        return m.to(device)

    def evaluate(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def run_config(lr, unfreeze_layer4, epochs):
        model = build_model(unfreeze_layer4)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
        val_acc = evaluate(model, val_loader)
        return model, val_acc

    # small hyperparameter search: learning rate x how much of the backbone to fine-tune
    search_space = [
        {"lr": 1e-3, "unfreeze_layer4": False, "epochs": 8},
        {"lr": 3e-4, "unfreeze_layer4": True, "epochs": 8},
        {"lr": 1e-4, "unfreeze_layer4": True, "epochs": 12},
    ]

    best_model, best_acc, best_cfg = None, -1, None
    for cfg in search_space:
        t0 = time.time()
        model, val_acc = run_config(cfg["lr"], cfg["unfreeze_layer4"], cfg["epochs"])
        print(f"  config {cfg} -> val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")
        if val_acc > best_acc:
            best_model, best_acc, best_cfg = model, val_acc, cfg

    print(f"  best config: {best_cfg}  val_acc={best_acc:.4f}")
    assert best_acc >= ACCEPTED_ACCURACY["images"], "image model below accepted accuracy"

    torch.save(best_model.state_dict(), MODELS_DIR / "image_model.pt")
    with open(MODELS_DIR / "image_classes.json", "w") as f:
        json.dump({"classes": class_names, "unfreeze_layer4": best_cfg["unfreeze_layer4"]}, f)
    print("  saved models/image_model.pt + models/image_classes.json")
    return best_acc


if __name__ == "__main__":
    results = {
        "classification": train_classification(),
        "nlp": train_nlp(),
        "sentiment": train_sentiment(),
        "images": train_images(),
    }
    print("\n=== Final test accuracies ===")
    for k, v in results.items():
        print(f"  {k:15s}: {v:.4f}  (min accepted {ACCEPTED_ACCURACY[k]})")
