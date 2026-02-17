"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.visualization.phase_diagrams — Phase Diagram Visualizations
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from qgt.visualization.styles import set_academic_style, COLORS, CMAPS


class PhaseDiagramPlotter:
    """
    Phase diagram overlays combining geometric and physical quantities.
    
    Parameters
    ----------
    output_dir : str
    """
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        set_academic_style()
    
    def plot_phase_diagram(self, qgt_data: dict,
                            gap_data: dict = None,
                            geodesic_paths: list = None,
                            model_name: str = 'TFIM',
                            save_name: str = 'phase_diagram'):
        """
        Combined phase diagram with metric, gap, phase labels, and geodesics.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Background: log metric determinant
        Z = np.log10(np.abs(qgt_data['metric_det']) + 1e-15)
        im = ax.pcolormesh(qgt_data['param_1'], qgt_data['param_0'], Z,
                            cmap='inferno', shading='gouraud', alpha=0.8)
        fig.colorbar(im, ax=ax, label='$\\log_{10}\\det(g)$')
        
        # Critical line
        lim = [min(qgt_data['param_0'].min(), qgt_data['param_1'].min()),
               max(qgt_data['param_0'].max(), qgt_data['param_1'].max())]
        ax.plot(lim, lim, '--', color='white', linewidth=2.5,
                 label='Critical: $h = J$')
        
        # Phase labels
        mid0 = np.mean(qgt_data['param_0'][[0, -1]])
        mid1 = np.mean(qgt_data['param_1'][[0, -1]])
        ax.text(mid1 * 0.5, mid0 * 1.5, 'Ferromagnetic\n$\\langle Z \\rangle \\neq 0$',
                 fontsize=14, color='white', ha='center', va='center',
                 fontweight='bold')
        ax.text(mid1 * 1.5, mid0 * 0.5, 'Paramagnetic\n$\\langle Z \\rangle = 0$',
                 fontsize=14, color='white', ha='center', va='center',
                 fontweight='bold')
        
        # Geodesics
        if geodesic_paths:
            colors = COLORS['geodesic']
            for i, path in enumerate(geodesic_paths):
                c = colors[i % len(colors)]
                ax.plot(path[:, 1], path[:, 0], color=c,
                         linewidth=2.5, alpha=0.9)
        
        ax.set_xlabel('Transverse field $h$', fontsize=14)
        ax.set_ylabel('Coupling $J$', fontsize=14)
        ax.set_title(f'{model_name} — Geometric Phase Diagram', fontsize=16)
        ax.legend(loc='upper right', fontsize=12)
        
        fig.savefig(f"{self.output_dir}/{save_name}.png")
        return fig
    
    def plot_scaling_analysis(self, scaling_data: dict,
                               save_name: str = 'finite_size_scaling'):
        """
        Plot finite-size scaling of fidelity susceptibility peaks.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sizes = np.array(scaling_data['sizes'])
        peaks = np.array(scaling_data['peak_heights'])
        positions = np.array(scaling_data['peak_positions'])
        
        # Peak height scaling
        axes[0].loglog(sizes, peaks, 'o-', color=COLORS['primary'],
                         markersize=8, linewidth=2)
        axes[0].set_xlabel('System size $N$')
        axes[0].set_ylabel('$\\chi_F^{\\mathrm{peak}}$')
        axes[0].set_title('Peak Height Scaling')
        
        # Peak position convergence
        axes[1].plot(1/sizes, positions, 'o-', color=COLORS['secondary'],
                      markersize=8, linewidth=2)
        axes[1].set_xlabel('$1/N$')
        axes[1].set_ylabel('$h_c(N)$')
        axes[1].set_title('Critical Point Convergence')
        axes[1].axhline(y=1.0, linestyle='--', color=COLORS['critical'],
                          label='$h_c^{\\infty} = 1$')
        axes[1].legend()
        
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/{save_name}.png")
        return fig
