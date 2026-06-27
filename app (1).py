from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import uuid
import base64

app = Flask(__name__)

# ----------------------------
# Load model & classes
# ----------------------------
MODEL_PATH = "plant_disease_model.h5"
CLASSES_PATH = "classes.npy"

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

if os.path.exists(CLASSES_PATH):
    classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
    print(f"✅ Loaded {len(classes)} classes from {CLASSES_PATH}")
else:
    num_classes = model.output_shape[-1]
    classes = [f"Class_{i}" for i in range(num_classes)]
    print(f"⚠️ No classes file found. Using {num_classes} placeholder class names.")

# ----------------------------
# Upload folder
# ----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# Homepage
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# Prediction
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    filename = None
    filepath = None

    # Case 1: User uploaded a file
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

    # Case 2: User captured image from camera
    elif "captured_image" in request.form and request.form["captured_image"] != "":
        data_url = request.form["captured_image"]
        header, encoded = data_url.split(",", 1)
        img_data = base64.b64decode(encoded)
        filename = str(uuid.uuid4()) + "_captured.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        with open(filepath, "wb") as f:
            f.write(img_data)

    else:
        return redirect(url_for("home"))

    # Preprocess image
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict
    pred = model.predict(img_resized)
    class_idx = np.argmax(pred)
    confidence = np.max(pred) * 100
    label = classes[class_idx]

    return render_template("index.html", filename=filename, label=label, confidence=confidence)

# ----------------------------
# Serve uploaded images
# ----------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
