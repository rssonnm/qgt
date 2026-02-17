#!/usr/bin/env python3
"""
Phát hiện chuyển pha lượng tử trong mô hình Ising qua hình học QGT.

Quét fidelity susceptibility χ_F, metric determinant, và energy gap
dọc theo h/J, cho thấy đỉnh (singularity) tại h_c = J.
Tăng số site để thấy finite-size scaling.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgt.models.ising import IsingModel
from qgt.core.qgt import QuantumGeometricTensor
from qgt.visualization.styles import set_academic_style, COLORS


def critical_point_scan(N, n_pts=50):
    """Quét χ_F và Δ dọc theo h, cố định J=1."""
    model = IsingModel(n_sites=N, periodic=True)
    qgt = QuantumGeometricTensor(model.ground_state, n_params=2)

    h_vals = np.linspace(0.1, 2.0, n_pts)
    chi_F = np.zeros(n_pts)
    gap = np.zeros(n_pts)

    for i, h in enumerate(h_vals):
        params = np.array([1.0, h])
        g = qgt.metric(params)
        chi_F[i] = np.trace(g) / N
        gap[i] = model.energy_gap(params)

    idx_peak = np.argmax(chi_F)
    return {
        'h': h_vals, 'chi_F': chi_F, 'gap': gap,
        'h_c': h_vals[idx_peak], 'chi_peak': chi_F[idx_peak],
    }


def main():
    set_academic_style()
    outdir = 'output/examples'
    os.makedirs(outdir, exist_ok=True)

    print("── Chuyển pha lượng tử trong Ising TFIM ────────")
    print("   H = −J ΣZ_iZ_{i+1} − h ΣX_i")
    print("   Lý thuyết: h_c / J = 1\n")

    # Finite-size scaling
    sizes = [4, 6, 8]
    results = {}

    for N in sizes:
        t0 = time.time()
        results[N] = critical_point_scan(N, n_pts=40)
        dt = time.time() - t0
        print(f"  N={N}: h_c = {results[N]['h_c']:.3f}, "
              f"χ_F^peak = {results[N]['chi_peak']:.4f}  ({dt:.1f}s)")

    # Vẽ
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for N in sizes:
        r = results[N]
        axes[0].plot(r['h'], r['chi_F'], linewidth=2, label=f'N={N}')
        axes[1].plot(r['h'], r['gap'], linewidth=2, label=f'N={N}')

    axes[0].axvline(1.0, color='grey', linestyle='--', alpha=0.6)
    axes[0].set_xlabel('h / J')
    axes[0].set_ylabel('χ_F (per site)')
    axes[0].set_title('Fidelity Susceptibility')
    axes[0].legend()

    axes[1].axvline(1.0, color='grey', linestyle='--', alpha=0.6)
    axes[1].set_xlabel('h / J')
    axes[1].set_ylabel('Δ')
    axes[1].set_title('Energy Gap')
    axes[1].legend()

    # Scaling plot
    Ns = np.array(sizes)
    peaks = np.array([results[N]['chi_peak'] for N in sizes])
    axes[2].plot(Ns, peaks, 'o-', color=COLORS['primary'],
                  markersize=8, linewidth=2)
    axes[2].set_xlabel('N (system size)')
    axes[2].set_ylabel('χ_F^peak')
    axes[2].set_title('Peak Scaling')

    plt.tight_layout()
    fig.savefig(f'{outdir}/ising_qpt_scaling.png', dpi=200, bbox_inches='tight')
    print(f"\nPlot: {outdir}/ising_qpt_scaling.png")
    print("\nKết luận: χ_F đạt đỉnh tại h ≈ J, hội tụ về h_c = 1 khi N → ∞. ✓")


if __name__ == "__main__":
    main()
