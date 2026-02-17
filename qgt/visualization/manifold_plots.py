"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.visualization.manifold_plots — Information Manifold Visualizations
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

from qgt.visualization.styles import set_academic_style, COLORS, CMAPS


class ManifoldPlotter:
    """
    Publication-quality plots for QGT and information manifold data.
    
    Parameters
    ----------
    output_dir : str
        Directory for saving plots.
    """
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        set_academic_style()
    
    def plot_heatmap(self, data: dict, quantity: str,
                      title: str = None,
                      param_labels: tuple = ('$J$', '$h$'),
                      cmap: str = None,
                      log_scale: bool = False,
                      save_name: str = None,
                      ax: Optional[plt.Axes] = None):
        """
        Plot a 2D heatmap of a geometric quantity.
        
        Parameters
        ----------
        data : dict
            Must have 'param_0', 'param_1', and the quantity key.
        quantity : str
            Key in data dict.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()
        
        Z = data[quantity]
        if log_scale:
            Z = np.log10(np.abs(Z) + 1e-15)
        
        if cmap is None:
            cmap = CMAPS.get(quantity, 'inferno')
        
        im = ax.pcolormesh(data['param_1'], data['param_0'], Z,
                            cmap=cmap, shading='gouraud')
        cbar = fig.colorbar(im, ax=ax)
        
        ax.set_xlabel(param_labels[1])
        ax.set_ylabel(param_labels[0])
        ax.set_title(title or quantity.replace('_', ' ').title())
        
        if save_name:
            fig.savefig(f"{self.output_dir}/{save_name}.png")
        
        return fig, ax
    
    def plot_overview(self, qgt_data: dict,
                       save_name: str = 'manifold_overview'):
        """
        4-panel overview: metric det, fidelity sus, Berry curv, metric trace.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        panels = [
            ('metric_det', 'Metric Determinant det($g$)', True),
            ('fidelity_sus', 'Fidelity Susceptibility $\\chi_F$', True),
            ('berry_curv', 'Berry Curvature $\\Omega_{12}$', False),
            ('metric_trace', 'Metric Trace Tr($g$)', True),
        ]
        
        for ax, (key, title, log_sc) in zip(axes.ravel(), panels):
            if key in qgt_data:
                self.plot_heatmap(qgt_data, key, title,
                                   log_scale=log_sc, ax=ax)
        
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/{save_name}.png")
        return fig
    
    def plot_geodesics(self, geodesic_paths: List[np.ndarray],
                        background_data: dict = None,
                        background_key: str = 'metric_det',
                        critical_line: bool = True,
                        save_name: str = 'geodesics'):
        """
        Plot geodesic paths on the parameter space.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        
        if background_data is not None and background_key in background_data:
            Z = np.log10(np.abs(background_data[background_key]) + 1e-15)
            ax.pcolormesh(background_data['param_1'],
                           background_data['param_0'], Z,
                           cmap='Greys', alpha=0.3, shading='gouraud')
        
        colors = COLORS['geodesic']
        for i, path in enumerate(geodesic_paths):
            color = colors[i % len(colors)]
            ax.plot(path[:, 1], path[:, 0], color=color,
                     linewidth=2.5, alpha=0.9)
            ax.plot(path[0, 1], path[0, 0], 'o', color=color,
                     markersize=8, zorder=5)
        
        if critical_line:
            lim = ax.get_xlim()
            vals = np.linspace(*lim, 100)
            ax.plot(vals, vals, '--', color=COLORS['critical'],
                     linewidth=2, label='Critical line ($h = J$)')
            ax.legend()
        
        ax.set_xlabel('$h$')
        ax.set_ylabel('$J$')
        ax.set_title('Geodesics on Information Manifold')
        fig.savefig(f"{self.output_dir}/{save_name}.png")
        return fig
    
    def plot_cross_section(self, data: dict,
                            fixed_label: str = '$J = 1$',
                            scan_label: str = '$h$',
                            save_name: str = 'cross_section'):
        """
        Plot 1D cross-section of geometric quantities.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        x = data['scan_values']
        
        panels = [
            (axes[0, 0], 'fidelity_sus', 'Fidelity Susceptibility $\\chi_F$'),
            (axes[0, 1], 'metric_det', 'Metric Determinant'),
            (axes[1, 0], 'berry_curv', 'Berry Curvature $\\Omega_{12}$'),
            (axes[1, 1], 'energy_gap', 'Energy Gap $\\Delta$'),
        ]
        
        for ax, key, title in panels:
            if key in data:
                ax.plot(x, data[key], color=COLORS['primary'], linewidth=2)
                ax.axvline(x=1.0, color=COLORS['critical'], linestyle='--',
                            alpha=0.7, label='$h_c$')
                ax.set_xlabel(scan_label)
                ax.set_title(title)
                ax.legend()
        
        fig.suptitle(f'Cross-Section at {fixed_label}', fontsize=15, y=1.02)
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/{save_name}.png")
        return fig
