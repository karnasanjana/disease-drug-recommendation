"""Flask web app: users sign up, log in, and describe symptoms (typed or spoken)
to a chatbot that predicts the disease with a CNN and suggests diet and medication."""
import os
import pickle
import sqlite3
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from tensorflow import keras
from werkzeug.security import check_password_hash, generate_password_hash

from preprocess import clean_text
from recommendations import display_name, get_diet, get_drugs

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DB_PATH = BASE_DIR / "users.db"
LOW_CONFIDENCE = 0.5

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

if not (MODEL_DIR / "disease_cnn.keras").exists():
    raise SystemExit("No trained model found. Run `python train.py` first.")
model = keras.models.load_model(MODEL_DIR / "disease_cnn.keras")
with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
    _artifacts = pickle.load(f)
vectorizer, labels = _artifacts["vectorizer"], _artifacts["labels"]


def get_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, name TEXT, email TEXT, mobile TEXT,
        password_hash TEXT NOT NULL)""")
    return con


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/logon")
def logon():
    return render_template("signup.html")


@app.route("/login")
def login():
    return render_template("signin.html")


@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("user", "").strip()
    password = request.form.get("password", "")
    if not username or not password:
        return render_template("signup.html", error="Username and password are required.")
    try:
        with get_db() as con:
            con.execute(
                "INSERT INTO users (username, name, email, mobile, password_hash) VALUES (?, ?, ?, ?, ?)",
                (username, request.form.get("name", ""), request.form.get("email", ""),
                 request.form.get("mobile", ""), generate_password_hash(password)))
    except sqlite3.IntegrityError:
        return render_template("signup.html", error="That username is already taken.")
    return render_template("signin.html", message="Account created. Please sign in.")


@app.route("/signin", methods=["POST"])
def signin():
    username = request.form.get("user", "").strip()
    with get_db() as con:
        row = con.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if row is None or not check_password_hash(row[0], request.form.get("password", "")):
        return render_template("signin.html", error="Invalid username or password.")
    session["user"] = username
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/index")
@login_required
def index():
    return render_template("index.html")


@app.route("/predict")
@login_required
def predict():
    text = clean_text(request.args.get("mytext", ""))
    if not text:
        return jsonify({"response": "Please describe your symptoms, e.g. 'itching skin rash'."})

    features = vectorizer.transform([text]).toarray()[..., None, None]
    probs = model.predict(features, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    label = labels[int(top3[0])]
    confidence = float(probs[top3[0]])
    alternatives = ", ".join(f"{display_name(labels[i])} ({probs[i]:.0%})" for i in top3)
    drugs = get_drugs(label)

    response = f"Disease predicted: {display_name(label)} ({confidence:.0%} confidence)\n"
    if confidence < LOW_CONFIDENCE:
        response += "Low confidence - please describe more symptoms for a reliable prediction.\n"
    response += (f"Top candidates: {alternatives}\n\n"
                 f"Diet: {get_diet(label)}\n\n"
                 f"Medication: {', '.join(drugs) if drugs else 'Consult a doctor.'}\n\n"
                 "This is an academic project, not medical advice. Please consult a doctor.")
    return jsonify({
        "disease": display_name(label),
        "confidence": round(confidence, 3),
        "top_candidates": [{"disease": display_name(labels[i]), "probability": round(float(probs[i]), 3)} for i in top3],
        "response": response,
    })


@app.route("/note")
@login_required
def note():
    return render_template("Notebook.html")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
