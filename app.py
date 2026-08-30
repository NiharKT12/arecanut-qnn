import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, render_template, request

from qnn_numpy import HybridModel

# -----------------------
# Load Model
# -----------------------

model = HybridModel(Path(__file__).with_name("weights.npz"))


# -----------------------
# Transform
# -----------------------

def transform(image):
    """Resize((128,128)) + ToTensor() + Normalize([0.5]*3, [0.5]*3)."""
    image = image.resize((128, 128), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (array - 0.5) / 0.5


# -----------------------
# Prediction
# -----------------------

classes = ["Healthy", "Yellow Leaf"]


def predict(image):
    output = model(transform(image))
    return classes[int(output.argmax())]


# -----------------------
# Flask UI
# -----------------------

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    image_preview = None
    image_title = None

    if request.method == "POST":
        file = request.files.get("leaf_image")

        if file is None or file.filename == "":
            error = "Please choose an image file before submitting."
        elif not allowed_file(file.filename):
            error = "Only .jpg, .jpeg, and .png files are supported."
        else:
            try:
                image = Image.open(file.stream).convert("RGB")
                image_title = file.filename
                preview_buffer = io.BytesIO()
                image.save(preview_buffer, format="PNG")
                image_preview = "data:image/png;base64," + base64.b64encode(
                    preview_buffer.getvalue()
                ).decode("ascii")
                prediction = predict(image)
            except Exception:
                error = "Unable to process this image. Please upload a valid image file."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        image_preview=image_preview,
        image_title=image_title,
    )


if __name__ == "__main__":
    app.run(debug=True)
