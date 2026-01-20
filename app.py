from flask import Flask, render_template, request, redirect, session
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

import os
import sqlite3
import numpy as np

from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image

# ======================
# CONFIG FLASK
# ======================
app = Flask(__name__)
app.secret_key = "secretkey123"

UPLOAD_FOLDER = "uploads/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# ======================
# DATABASE SQLITE
# ======================
def get_db():
    return sqlite3.connect("users.db")


with get_db() as conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """
    )

# ======================
# LOAD MODEL AI
# ======================
model = MobileNetV2(weights="imagenet")


# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()

        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect("/dashboard")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password),
            )
            conn.commit()
            return redirect("/")
        except:
            pass

    return render_template("register.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/")

    predictions = None

    # =========================
    # STEP 3: PROSES UPLOAD & SIMPAN HISTORY
    # =========================
    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)[0]

        predictions = [
            {"breed": name, "confidence": float(round(prob * 100, 2))}
            for (_, name, prob) in decoded
        ]

        # SIMPAN KE FIRESTORE
        db.collection("history").add(
            {
                "username": session["user"],
                "image_path": filepath,
                "predictions": predictions,
                "created_at": datetime.now(),
            }
        )

    # =========================
    # STEP 4: AMBIL HISTORY USER
    # =========================
    history_ref = (
        db.collection("history")
        .where("username", "==", session["user"])
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .stream()
    )

    history = []
    for doc in history_ref:
        data = doc.to_dict()
        data["id"] = doc.id
        history.append(data)

    # =========================
    # KIRIM KE TEMPLATE
    # =========================
    return render_template("dashboard.html", predictions=predictions, history=history)


@app.route("/history/<doc_id>")
def load_history(doc_id):
    doc = db.collection("history").document(doc_id).get()
    if not doc.exists:
        return redirect("/dashboard")

    data = doc.to_dict()
    return render_template(
        "dashboard.html", predictions=data["predictions"], history=[]
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
