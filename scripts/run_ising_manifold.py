#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  run_ising_manifold.py — Full Ising Information Manifold Analysis
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python scripts/run_ising_manifold.py [--n_sites 4] [--n_points 25]
"""

import argparse
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgt.models.ising import IsingModel
from qgt.applications.information_manifold import InformationManifoldAnalyzer
from qgt.visualization.manifold_plots import ManifoldPlotter
from qgt.visualization.phase_diagrams import PhaseDiagramPlotter


def main():
    parser = argparse.ArgumentParser(description='Ising Information Manifold')
    parser.add_argument('--n_sites', type=int, default=4)
    parser.add_argument('--n_points', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default='output/ising')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"═══ Ising Information Manifold (N={args.n_sites}) ═══")
    
    # 1. Setup
    model = IsingModel(n_sites=args.n_sites)
    analyzer = InformationManifoldAnalyzer(model)
    print(f"Model: {model}")
    print(f"Critical point: {model.critical_point['condition']}")
    
    # 2. Full analysis
    param_ranges = [(0.1, 2.0), (0.1, 2.0)]
    results = analyzer.full_analysis(
        param_ranges=param_ranges,
        n_points=args.n_points,
        compute_curvature=True,
        compute_geodesics=True,
    )
    
    # 3. Cross-section at J=1
    cross = analyzer.cross_section(
        fixed_param_idx=0, fixed_value=1.0,
        scan_range=(0.1, 2.0), n_points=50
    )
    
    # 4. Visualize
    plotter = ManifoldPlotter(args.output_dir)
    plotter.plot_overview(results['qgt_scan'], save_name='overview')
    plotter.plot_geodesics(results.get('geodesics', []),
                            background_data=results['qgt_scan'])
    plotter.plot_cross_section(cross)
    
    phase_plotter = PhaseDiagramPlotter(args.output_dir)
    phase_plotter.plot_phase_diagram(
        results['qgt_scan'],
        geodesic_paths=results.get('geodesics', []),
    )
    
    # 5. Summary
    if results.get('chern_number') is not None:
        print(f"Chern number: {results['chern_number']:.4f}")
    print(f"Total time: {results['total_time']:.1f}s")
    print(f"Plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
