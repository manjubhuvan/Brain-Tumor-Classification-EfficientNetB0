from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename
from PIL import Image

# Flask setup
app = Flask(__name__)
app.secret_key = "replace-this"  

# static/uploads inside the app folder
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


MODEL_PATH = os.path.join(app.root_path, "best_efficientnetb0.h5")
model = load_model(MODEL_PATH, compile=False)

try:
    _h = int(model.inputs[0].shape[1])
    _w = int(model.inputs[0].shape[2])
    INPUT_SIZE = (_h, _w)
except Exception:
    INPUT_SIZE = (224, 224)

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type.")
        return redirect(url_for("index"))

    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    with Image.open(file_path) as im:
        im = im.convert("RGB")
        im = im.resize(INPUT_SIZE)
        im.save(file_path)  

    img = image.load_img(file_path, target_size=INPUT_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)
    if isinstance(preds, (list, tuple)):
        preds = preds[-1]
    pred_idx = int(np.argmax(preds[0]))
    pred_label = labels[pred_idx]
    confidence = float(preds[0][pred_idx]) * 100.0

    return render_template(
        "result.html",
        prediction=pred_label.upper(),
        confidence=f"{confidence:.2f}",
        image_url=url_for("static", filename=f"uploads/{filename}"),
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
