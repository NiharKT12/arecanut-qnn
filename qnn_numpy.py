"""Pure-NumPy inference for the hybrid CNN + 4-qubit QNN arecanut classifier.

This mirrors the original PyTorch/PennyLane model exactly (see notebook/arecanut.ipynb)
without pulling in torch, torchvision or pennylane, which together blew past the
500 MB serverless function limit.
"""

import numpy as np

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS


# -----------------------
# Classical layers
# -----------------------

def conv2d(x, weight, bias):
    """Valid 3x3 convolution, stride 1. x: (C, H, W) -> (F, H-2, W-2)."""
    f, c, kh, kw = weight.shape
    _, h, w = x.shape
    oh, ow = h - kh + 1, w - kw + 1

    # im2col: (C*kh*kw, oh*ow)
    cols = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))
    cols = cols.transpose(0, 3, 4, 1, 2).reshape(c * kh * kw, oh * ow)

    out = weight.reshape(f, -1) @ cols + bias[:, None]
    return out.reshape(f, oh, ow)


def relu(x):
    return np.maximum(x, 0.0)


def maxpool2(x):
    """MaxPool2d(2) with the default stride of 2; trailing odd row/col is dropped."""
    c, h, w = x.shape
    h, w = h - h % 2, w - w % 2
    return x[:, :h, :w].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def linear(x, weight, bias):
    return weight @ x + bias


# -----------------------
# Quantum circuit
# -----------------------

def _apply_1q(state, gate, wire):
    """Apply a 2x2 gate to `wire` of a (2,)*N_QUBITS state tensor."""
    state = np.moveaxis(state, wire, 0)
    shape = state.shape
    state = gate @ state.reshape(2, -1)
    return np.moveaxis(state.reshape(shape), 0, wire)


def _apply_cnot(state, control, target):
    state = np.moveaxis(state, (control, target), (0, 1))
    state = state.copy()
    state[1] = state[1][::-1]
    return np.moveaxis(state, (0, 1), (control, target))


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rot(phi, theta, omega):
    """PennyLane Rot(phi, theta, omega) = RZ(omega) @ RY(theta) @ RZ(phi)."""
    rz_phi = np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])
    rz_omega = np.diag([np.exp(-0.5j * omega), np.exp(0.5j * omega)])
    return rz_omega @ _ry(theta) @ rz_phi


def quantum_circuit(inputs, weights):
    """RY embedding + StronglyEntanglingLayers, returning <PauliZ> per wire."""
    state = np.zeros((N_STATES,), dtype=complex)
    state[0] = 1.0
    state = state.reshape((2,) * N_QUBITS)

    for i in range(N_QUBITS):
        state = _apply_1q(state, _ry(inputs[i]), i)

    n_layers = weights.shape[0]
    for layer in range(n_layers):
        for i in range(N_QUBITS):
            state = _apply_1q(state, _rot(*weights[layer, i]), i)

        # Default `ranges` of StronglyEntanglingLayers for a CNOT imprimitive.
        r = layer % (N_QUBITS - 1) + 1
        for i in range(N_QUBITS):
            state = _apply_cnot(state, i, (i + r) % N_QUBITS)

    probs = np.abs(state) ** 2
    signs = np.array([1.0, -1.0])
    return np.array([
        np.tensordot(probs, signs, axes=([i], [0])).sum()
        for i in range(N_QUBITS)
    ])


# -----------------------
# Full forward pass
# -----------------------

class HybridModel:

    def __init__(self, weights_path):
        self.w = {k: v for k, v in np.load(weights_path).items()}

    def __call__(self, x):
        """x: (3, 128, 128) float32 -> (2,) logits."""
        w = self.w

        x = maxpool2(relu(conv2d(x, w["conv.0.weight"], w["conv.0.bias"])))
        x = maxpool2(relu(conv2d(x, w["conv.3.weight"], w["conv.3.bias"])))
        x = x.mean(axis=(1, 2))  # AdaptiveAvgPool2d((1, 1)) + Flatten

        x = linear(x, w["fc1.weight"], w["fc1.bias"])
        x = quantum_circuit(x, w["qnn.weights"])
        return linear(x, w["fc2.weight"], w["fc2.bias"])
