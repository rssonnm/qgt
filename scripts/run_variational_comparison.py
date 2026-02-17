#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  run_variational_comparison.py — Exact vs Variational Geometry
═══════════════════════════════════════════════════════════════════════════════

Compares the geometry of the exact ground state manifold with a
variational approximation using PennyLane VQE.

Usage:
    python scripts/run_variational_comparison.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgt.models.ising import IsingModel
from qgt.core.qgt import QuantumGeometricTensor
from qgt.applications.phase_transitions import PhaseTransitionDetector

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    output_dir = 'output/variational'
    os.makedirs(output_dir, exist_ok=True)
    
    print("═══ Exact vs Variational Geometry ═══")
    
    # 1. Exact analysis
    model = IsingModel(n_sites=4)
    detector = PhaseTransitionDetector(model)
    
    result = detector.find_critical_points(
        scan_param=1,
        scan_range=(0.1, 2.0),
        fixed_params={0: 1.0},
        n_points=40,
        metric='fidelity_sus',
    )
    
    print(f"Detected critical point: h_c = {result['critical_point']:.3f}")
    print(f"Peak fidelity susceptibility: {result['peak_value']:.4f}")
    
    # 2. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result['scan_values'], result['metric_values'],
             'b-', linewidth=2, label='Exact (ED)')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7,
                label='$h_c = J$')
    ax.set_xlabel('$h$', fontsize=14)
    ax.set_ylabel('$\\chi_F$ (per site)', fontsize=14)
    ax.set_title('Fidelity Susceptibility — Critical Point Detection')
    ax.legend(fontsize=12)
    fig.savefig(f'{output_dir}/critical_point_detection.png', dpi=300,
                 bbox_inches='tight')
    print(f"Plot saved to {output_dir}/")


if __name__ == '__main__':
    main()
