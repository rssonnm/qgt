#!/usr/bin/env python3
"""
Tính toán chính xác Quantum Geometric Tensor từ Jacobian trạng thái lượng tử.

Xây dựng QGT trực tiếp qua công thức:
    χ_ij = ⟨∂_i ψ | ∂_j ψ⟩ − ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩

Phương pháp: lấy Jacobian ∂ψ/∂θ bằng auto-differentiation,
rồi tính tích vô hướng trong không gian Hilbert.
Kết quả so sánh với qml.metric_tensor() của PennyLane.
"""

import pennylane as qml
from pennylane import numpy as np
import time

# ── Thiết bị ────────────────────────────────────────────────
dev = qml.device("default.qubit", wires=2)
dev_mt = qml.device("lightning.qubit", wires=3)  # extra wire for metric_tensor


@qml.qnode(dev, diff_method="backprop")
def ansatz_state(params):
    """Mạch tham số hóa 2-qubit → trả về trạng thái |ψ(θ)⟩."""
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=0)
    return qml.state()


@qml.qnode(dev_mt)
def ansatz_expval(params):
    """Cùng mạch, trả về ⟨Z₀⟩ (dùng cho metric_tensor)."""
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=0)
    return qml.expval(qml.PauliZ(0))


def qgt_from_jacobian(params):
    """
    Xây dựng toàn bộ QGT từ Jacobian.

    Bước 1: Tính ψ = |ψ(θ)⟩
    Bước 2: Tính J_kα = ∂ψ_k/∂θ_α  (Jacobian phức)
    Bước 3: χ_αβ = (J†J)_αβ − (J†ψ)_α (ψ†J)_β
    """
    psi = ansatz_state(params)

    # Autograd backprop cho ra Jacobian real/imag tách rời
    def split_real_imag(p):
        s = ansatz_state(p)
        return np.hstack([np.real(s), np.imag(s)])

    J_ri = qml.jacobian(split_real_imag)(params)
    dim = len(psi)
    J = J_ri[:dim, :] + 1j * J_ri[dim:, :]   # (dim, n_params) phức

    # ── Tính QGT ──
    overlap_matrix = np.conj(J.T) @ J                   # ⟨∂_i ψ|∂_j ψ⟩
    proj_vec = np.conj(J.T) @ psi                       # ⟨∂_i ψ|ψ⟩
    projection = np.outer(proj_vec, np.conj(proj_vec))   # loại bỏ thành phần song song

    return overlap_matrix - projection


def main():
    theta = np.array([0.5, 0.1, 0.8], requires_grad=True)

    print("── QGT từ State Jacobian ────────────────────────")
    print(f"θ = {theta}\n")

    t0 = time.time()
    chi = qgt_from_jacobian(theta)
    dt = time.time() - t0

    g = np.real(chi)
    omega = 2 * np.imag(chi)

    print(f"QGT χ_αβ (tính trong {dt:.4f}s):")
    print(chi, "\n")
    print("Fubini-Study metric g_αβ = Re(χ):")
    print(g, "\n")
    print("Berry curvature Ω_αβ = 2·Im(χ):")
    print(omega, "\n")

    # ── So sánh với PennyLane ──
    print("── PennyLane qml.metric_tensor ──────────────────")
    t0 = time.time()
    g_pl = qml.metric_tensor(ansatz_expval)(theta)
    dt_pl = time.time() - t0
    print(f"g_PL (tính trong {dt_pl:.4f}s):")
    print(np.array(g_pl), "\n")

    # ── Kiểm tra ──
    diff = np.max(np.abs(g - g_pl))
    print(f"Sai lệch tối đa: {diff:.2e}")
    assert diff < 1e-5, "Kết quả không khớp!"
    print("Kết quả khớp. ✓")


if __name__ == "__main__":
    main()
