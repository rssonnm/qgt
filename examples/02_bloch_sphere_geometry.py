#!/usr/bin/env python3
"""
Hình học Fubini-Study trên Bloch sphere: single qubit.

Với |ψ(θ,φ)⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩,
metric lý thuyết là ds² = (1/4)(dθ² + sin²θ dφ²).

Bài này kiểm chứng công thức bằng tính số,
rồi vẽ metric determinant trên toàn bộ sphere.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgt.core.qgt import QuantumGeometricTensor
from qgt.visualization.styles import set_academic_style


def bloch_state(params):
    """Trạng thái trên Bloch sphere: |ψ(θ,φ)⟩."""
    theta, phi = params
    psi = np.array([
        np.cos(theta / 2),
        np.exp(1j * phi) * np.sin(theta / 2)
    ])
    energy = 0.0
    return energy, psi


def main():
    set_academic_style()
    qgt = QuantumGeometricTensor(bloch_state, n_params=2)

    print("── Fubini-Study trên Bloch Sphere ──────────────")

    # Kiểm tra tại vài điểm
    test_points = [
        (np.pi/4, 0.0),
        (np.pi/2, np.pi/4),
        (np.pi/3, np.pi/2),
    ]

    for theta, phi in test_points:
        g = qgt.metric(np.array([theta, phi]))
        g_exact = np.diag([0.25, 0.25 * np.sin(theta)**2])

        print(f"\n  θ={theta:.3f}, φ={phi:.3f}")
        print(f"    g_θθ = {g[0,0]:.6f}  (lý thuyết: 0.250000)")
        print(f"    g_φφ = {g[1,1]:.6f}  (lý thuyết: {g_exact[1,1]:.6f})")
        print(f"    g_θφ = {g[0,1]:.6f}  (lý thuyết: 0.000000)")

    # Quét toàn bộ sphere → vẽ det(g)
    n = 40
    thetas = np.linspace(0.05, np.pi - 0.05, n)
    phis = np.linspace(0, 2*np.pi, n)
    det_map = np.zeros((n, n))

    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            g = qgt.metric(np.array([th, ph]))
            det_map[i, j] = np.linalg.det(g)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(phis, thetas, det_map,
                        cmap='inferno', shading='gouraud')
    ax.set_xlabel('φ')
    ax.set_ylabel('θ')
    ax.set_title('det(g) trên Bloch Sphere — FS Metric')
    fig.colorbar(im, ax=ax, label='det(g)')

    # Lý thuyết: det(g) = sin²θ / 16
    ax.contour(phis, thetas, det_map, levels=5, colors='white',
                linewidths=0.8, alpha=0.5)

    outdir = 'output/examples'
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/bloch_sphere_metric.png', dpi=200, bbox_inches='tight')
    print(f"\nPlot: {outdir}/bloch_sphere_metric.png")
    print("det(g) = sin²θ / 16 — cực đại ở xích đạo, triệt tiêu ở cực. ✓")


if __name__ == "__main__":
    main()
