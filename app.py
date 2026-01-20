from flask import Flask, render_template, request, redirect, session
from firebase_admin import credentials, firestore
from datetime import datetime
from flask import send_from_directory
import firebase_admin

import cv2
import base64
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

FACE_FOLDER = "faces"
os.makedirs(FACE_FOLDER, exist_ok=True)

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
# Face Histogram
# ======================
def extract_face_histogram(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten().tolist()

def compare_histograms(hist1, hist2):
    hist1 = np.array(hist1, dtype=np.float32)
    hist2 = np.array(hist2, dtype=np.float32)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score


def save_image_with_fallback(file):
    filename = file.filename

    # ====== PATH LOCAL ======
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        # ====== SIMULASI FIREBASE STORAGE ======
        # (anggap ini Firebase Storage)
        raise Exception("Firebase Storage unavailable")

    except Exception as e:
        # ====== FALLBACK KE LOCAL ======
        file.save(local_path)
        return {
            "storage": "local",
            "path": local_path
        }
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


@app.route("/login-face", methods=["POST"])
def login_face():
    username = request.form["username"]
    face_file = request.files["face_image"]

    user_doc = db.collection("users").document(username).get()
    if not user_doc.exists:
        return redirect("/")

    user_data = user_doc.to_dict()

    if user_data.get("face_histogram") is None:
        return redirect("/")

    face_path = "temp_face.jpg"
    face_file.save(face_path)

    input_hist = extract_face_histogram(face_path)
    stored_hist = user_data["face_histogram"]

    score = compare_histograms(input_hist, stored_hist)
    
    if os.path.exists(face_path):
        os.remove(face_path)

    if score > 0.75:
        session["user"] = username
        return redirect("/dashboard")

    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        
        face_hist = None

        if "face_image" in request.files:
            face_file = request.files["face_image"]

            if face_file.filename != "":    
                face_path = os.path.join(FACE_FOLDER, f"{username}.jpg")
                face_file.save(face_path)

                face_hist = extract_face_histogram(face_path)
                
                if os.path.exists(face_path):
                    os.remove(face_path)
    
        db.collection("users").document(username).set({
            "username": username,
            "password": password,
            "face_histogram": face_hist
        })

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

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)

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

        save_result = save_image_with_fallback(file)
        filepath = save_result["path"]
        storage_type = save_result["storage"]

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
        db.collection("history").add({
            "username": session["user"],
            "image_path": filepath,
            "storage_type": storage_type,
            "predictions": predictions,
            "created_at": datetime.now(),
        })

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
        "dashboard.html",
        predictions=data["predictions"],
        history=[],
        data=data  # ⬅ kirim seluruh dokumen
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
