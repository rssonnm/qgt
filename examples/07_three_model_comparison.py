#!/usr/bin/env python3
"""
So sánh phase diagram hình học của 3 mô hình: Ising, Heisenberg, J1-J2.

Mỗi mô hình có đa tạp thông tin riêng với cấu trúc hình học khác nhau.
Quét fidelity susceptibility trên không gian tham số 2D,
cho thấy vị trí chuyển pha đặc trưng của từng mô hình.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgt.models.ising import IsingModel
from qgt.models.heisenberg import HeisenbergModel
from qgt.models.su2_model import SU2Model
from qgt.core.qgt import QuantumGeometricTensor
from qgt.visualization.styles import set_academic_style


def scan_chi_F(state_fn, n_params, n_sites, param_ranges, n_pts=20):
    """Quét fidelity susceptibility trên lưới 2D."""
    qgt = QuantumGeometricTensor(state_fn, n_params)
    p0 = np.linspace(*param_ranges[0], n_pts)
    p1 = np.linspace(*param_ranges[1], n_pts)
    chi = np.zeros((n_pts, n_pts))
    for i, x in enumerate(p0):
        for j, y in enumerate(p1):
            try:
                g = qgt.metric(np.array([x, y]))
                chi[i, j] = np.trace(g) / n_sites
            except Exception:
                chi[i, j] = np.nan
    return p0, p1, chi


def main():
    set_academic_style()
    outdir = 'output/examples'
    os.makedirs(outdir, exist_ok=True)

    N = 4
    n_pts = 18

    print("── Phase Diagrams: Ising vs Heisenberg vs J1-J2 ─")

    # ── Ising (J, h) ──
    print("\n  [1/3] Ising TFIM...")
    t0 = time.time()
    ising = IsingModel(n_sites=N)
    p0_i, p1_i, chi_i = scan_chi_F(
        ising.ground_state, 2, N, [(0.2, 2.0), (0.2, 2.0)], n_pts)
    print(f"    Done in {time.time()-t0:.1f}s")

    # ── Heisenberg (J, Δ), h=0 cố định ──
    print("  [2/3] Heisenberg XXZ (J, Δ)...")
    t0 = time.time()
    heisen = HeisenbergModel(n_sites=N)
    # Wrapper: chỉ scan J, Δ (h=0)
    def heisen_gs_2param(params):
        return heisen.ground_state(np.array([params[0], params[1], 0.0]))
    p0_h, p1_h, chi_h = scan_chi_F(
        heisen_gs_2param, 2, N, [(0.2, 2.0), (-1.5, 2.0)], n_pts)
    print(f"    Done in {time.time()-t0:.1f}s")

    # ── J1-J2 SU(2) ──
    print("  [3/3] J1-J2 SU(2)...")
    t0 = time.time()
    su2 = SU2Model(n_sites=N)
    p0_s, p1_s, chi_s = scan_chi_F(
        su2.ground_state, 2, N, [(0.2, 2.0), (0.0, 1.2)], n_pts)
    print(f"    Done in {time.time()-t0:.1f}s")

    # ── Vẽ 3 panel ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    panels = [
        (axes[0], p0_i, p1_i, chi_i, '$J$', '$h$',
         'Ising TFIM', [(0.2, 2.0), (0.2, 2.0)]),
        (axes[1], p0_h, p1_h, chi_h, '$J$', '$\\Delta$',
         'Heisenberg XXZ', None),
        (axes[2], p0_s, p1_s, chi_s, '$J_1$', '$J_2$',
         'J1-J2 SU(2)', None),
    ]

    for ax, p0, p1, chi, xlabel, ylabel, title, crit in panels:
        Z = np.log10(chi + 1e-10)
        im = ax.pcolormesh(p1, p0, Z, cmap='inferno', shading='gouraud')
        fig.colorbar(im, ax=ax, label='$\\log_{10} \\chi_F$')
        ax.set_xlabel(ylabel, fontsize=12)
        ax.set_ylabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=13)

    # Đường chuyển pha lý thuyết
    axes[0].plot([0.2, 2.0], [0.2, 2.0], '--', color='white',
                  linewidth=2, label='$h = J$')
    axes[0].legend(fontsize=10)
    axes[1].axvline(1.0, color='white', linestyle='--', linewidth=2)
    axes[1].axvline(-1.0, color='cyan', linestyle='--', linewidth=2)

    plt.tight_layout()
    fig.savefig(f'{outdir}/three_model_comparison.png',
                 dpi=200, bbox_inches='tight')
    print(f"\nPlot: {outdir}/three_model_comparison.png")
    print("Ba mô hình, ba phase diagram, cùng một ngôn ngữ hình học. ✓")


if __name__ == "__main__":
    main()
