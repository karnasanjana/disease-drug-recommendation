"""Train the symptom -> disease CNN and save it to model/.

Run once before starting the app:  python train.py

The raw dataset repeats the same symptom rows many times (4,920 rows, 304 unique),
so rows are de-duplicated before the train/test split; otherwise identical rows
land in both sets and test accuracy is meaningless.
"""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from preprocess import clean_text

SEED = 42
MODEL_DIR = Path("model")
DOCS_DIR = Path("docs")


def build_model(n_features, n_classes):
    # TF-IDF vector treated as an (n_features x 1 x 1) "image" for 1x1 convolutions
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1, 1)),
        layers.Conv2D(32, (1, 1), activation="relu"),
        layers.Conv2D(32, (1, 1), activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(texts, y, n_classes, epochs=40):
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None)
    X = vectorizer.fit_transform(texts).toarray()
    model = build_model(X.shape[1], n_classes)
    model.fit(X[..., None, None], y, batch_size=8, epochs=epochs, verbose=0)
    return vectorizer, model


def main():
    keras.utils.set_random_seed(SEED)
    df = pd.read_csv("Dataset/dataset.csv", encoding="ISO-8859-1")
    df["text"] = df["Target"].map(clean_text)
    df = df.drop_duplicates(subset=["text", "Source"]).reset_index(drop=True)

    labels = sorted(df["Source"].unique())
    y = df["Source"].map(labels.index).to_numpy()
    print(f"{len(df)} unique symptom sets, {len(labels)} diseases")

    # 1) Honest evaluation on a held-out split
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], y, test_size=0.2, stratify=y, random_state=SEED)
    vectorizer, model = fit(X_tr, y_tr, len(labels))
    pred = model.predict(vectorizer.transform(X_te).toarray()[..., None, None], verbose=0).argmax(1)
    metrics = {
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "n_classes": len(labels),
    }
    print(json.dumps(metrics, indent=2))

    DOCS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, labels=range(len(labels)), display_labels=labels,
        xticks_rotation=90, colorbar=False, ax=ax)
    ax.set_title(f"Held-out confusion matrix (accuracy {metrics['test_accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=80)

    # 2) Final model trained on all unique rows for deployment
    vectorizer, model = fit(df["text"], y, len(labels))
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "disease_cnn.keras")
    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "labels": labels}, f)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
