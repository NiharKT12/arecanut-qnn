# 🌴 Arecanut Leaf Disease Detection using Hybrid Quantum Neural Network

This project detects **Arecanut Leaf Diseases** using a **Hybrid Classical + Quantum Neural Network (CNN + QNN)** and provides a **Streamlit Web Application** for real-time prediction.

The model currently classifies:

- ✅ Healthy Leaf  
- ⚠️ Yellow Leaf Disease  
- 🍂 Leaf Spot Disease *(Future Work)*

---

# 🚀 Features

- Hybrid **CNN + Quantum Neural Network**
- Real-time prediction using **Streamlit**
- High accuracy (**~96% Test Accuracy**)
- Checkpoint saving & resume training
- Clean and simple UI
- Expandable to multiple diseases

---

# 🧠 Model Architecture

Image → CNN → Feature Extraction → Quantum Neural Network → Classification

### Technologies Used

- PyTorch
- PennyLane (Quantum Machine Learning)
- Streamlit
- Python
- Torchvision
- PIL

---

# 📊 Model Performance

| Metric | Value |
|--------|------|
| Training Accuracy | 96.30% |
| Test Accuracy | **95.98%** |
| Model Type | Hybrid CNN + QNN |
| Dataset | Arecanut Leaf Images |

---

# 📁 Project Structure

```text
arecanut-qnn/
│
├── app.py
├── best_arecanut_qnn.pth
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/NiharKT12/arecanut-qnn.git
cd arecanut-qnn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

Run the Streamlit application locally:

```bash
streamlit run app.py
```

1. Open the provided local URL (usually `http://localhost:8501`) in your web browser.
2. Upload an image of an Arecanut leaf (`.jpg`, `.jpeg`, or `.png`).
3. The hybrid quantum model will process the image and instantly output the prediction: **Healthy** or **Yellow Leaf**.


