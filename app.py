import base64
import io
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pennylane as qml

from PIL import Image
from flask import Flask, render_template, request

# -----------------------
# Quantum Setup
# -----------------------

n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.templates.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


weight_shapes = {"weights": (3, n_qubits, 3)}

qnn = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)

# -----------------------
# Model
# -----------------------

class HybridModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32,4)

        self.qnn = qnn

        self.fc2 = nn.Linear(4,2)

    def forward(self,x):

        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)

        x = x.float()

        qnn_outputs = []

        for i in range(x.shape[0]):
            q_out = self.qnn(x[i])
            qnn_outputs.append(q_out)

        x = torch.stack(qnn_outputs)

        x = self.fc2(x)

        return x


# -----------------------
# Load Model
# -----------------------

device = torch.device("cpu")

model = HybridModel()

model.load_state_dict(
    torch.load(Path(__file__).with_name("best_arecanut_qnn.pth"), map_location=device)
)

model.eval()


# -----------------------
# Transform
# -----------------------

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


# -----------------------
# Prediction
# -----------------------

classes = ["Healthy", "Yellow Leaf"]

def predict(image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    _, predicted = torch.max(output,1)

    return classes[predicted.item()]


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
