import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pennylane as qml

from PIL import Image

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
    torch.load("best_arecanut_qnn.pth", map_location=device)
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
# Streamlit UI
# -----------------------

st.title("Arecanut Leaf Disease Detection")

st.write("Upload an Arecanut Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = predict(image)

    st.success(f"Prediction: {result}")