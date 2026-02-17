#!/usr/bin/env python3
"""
Geodesic trên đa tạp thông tin lượng tử: đường thẳng nhất trong không gian Hilbert.

Tính và vẽ geodesic trên không gian tham số (J, h) của mô hình Ising.
Geodesic bị lệch mạnh khi đi qua vùng chuyển pha (nơi curvature lớn),
tương tự ánh sáng bị bẻ cong bởi trường hấp dẫn trong GR.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgt.models.ising import IsingModel
from qgt.core.qgt import QuantumGeometricTensor
from qgt.geometry.metric import MetricTensor
from qgt.geometry.geodesics import GeodesicSolver
from qgt.visualization.styles import set_academic_style, COLORS


def main():
    set_academic_style()
    outdir = 'output/examples'
    os.makedirs(outdir, exist_ok=True)

    print("── Geodesic trên Information Manifold ──────────")

    model = IsingModel(n_sites=4)
    qgt = QuantumGeometricTensor(model.ground_state, n_params=2)
    metric = MetricTensor(qgt.metric)
    solver = GeodesicSolver(metric)

    # Pre-compute Christoffel trên lưới
    param_ranges = [(0.2, 1.8), (0.2, 1.8)]
    print("  Pre-computing Christoffel symbols trên lưới 12×12...")
    t0 = time.time()
    solver.precompute_grid(param_ranges, n_grid=12)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Bắn geodesic từ pha sắt từ (J lớn, h nhỏ)
    starts = [
        (np.array([0.4, 0.4]), np.array([0.1, 0.08]), 'Ferro → Para (trực tiếp)'),
        (np.array([0.4, 0.4]), np.array([0.05, 0.12]), 'Dọc theo h'),
        (np.array([1.5, 0.4]), np.array([-0.08, 0.1]), 'Từ sâu trong pha ferro'),
        (np.array([0.4, 1.5]), np.array([0.1, -0.03]), 'Từ pha thuận từ'),
    ]

    fig, ax = plt.subplots(figsize=(9, 8))

    # Nền: log(det(g))
    n_bg = 30
    p0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_bg)
    p1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_bg)
    det_g = np.zeros((n_bg, n_bg))
    for i, x in enumerate(p0):
        for j, y in enumerate(p1):
            g = qgt.metric(np.array([x, y]))
            det_g[i, j] = np.log10(abs(np.linalg.det(g)) + 1e-20)

    ax.pcolormesh(p1, p0, det_g, cmap='Greys', alpha=0.25, shading='gouraud')

    # Vẽ geodesic
    colors = COLORS['geodesic']
    for idx, (x0, v0, label) in enumerate(starts):
        path = solver.solve(x0, v0, t_span=(0, 5.0), n_steps=500)
        if len(path) < 3:
            print(f"  Skipping '{label}' — path too short ({len(path)} pts)")
            continue
        c = colors[idx % len(colors)]
        ax.plot(path[:, 1], path[:, 0], color=c, linewidth=2.5,
                 alpha=0.9, label=label)
        ax.plot(path[0, 1], path[0, 0], 'o', color=c, markersize=8)

    # Đường chuyển pha
    ax.plot([0.2, 1.8], [0.2, 1.8], '--', color='red',
             linewidth=2, alpha=0.7, label='h = J (critical)')

    ax.set_xlabel('h', fontsize=13)
    ax.set_ylabel('J', fontsize=13)
    ax.set_title('Geodesics trên Information Manifold — Ising Model', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(param_ranges[1])
    ax.set_ylim(param_ranges[0])

    fig.savefig(f'{outdir}/geodesics_ising.png', dpi=200, bbox_inches='tight')
    print(f"\nPlot: {outdir}/geodesics_ising.png")
    print("Geodesic bị lệch bởi singularity tại h = J — tương tự gravitational lensing. ✓")


if __name__ == "__main__":
    main()
