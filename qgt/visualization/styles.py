"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.visualization.styles — Academic Publication Style Config
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


# Academic color palette
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#e74c3c',
    'accent': '#3498db',
    'warning': '#f39c12',
    'success': '#27ae60',
    'ferro': '#2c3e50',
    'para': '#e74c3c',
    'critical': '#f39c12',
    'geodesic': ['#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#e74c3c'],
}

# Colormap for QGT quantities
CMAPS = {
    'metric': 'inferno',
    'curvature': 'RdBu_r',
    'berry': 'coolwarm',
    'entropy': 'viridis',
    'gap': 'plasma',
}


def set_academic_style():
    """
    Apply publication-quality matplotlib style.
    
    Uses serif fonts, proper sizing, and clean aesthetics.
    Call this once at the start of any plotting script.
    """
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        
        # Figure
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        # Axes
        'axes.linewidth': 1.2,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # Grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Legend
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        
        # Math
        'mathtext.fontset': 'cm',
    })


def create_figure(n_rows: int = 1, n_cols: int = 1,
                   figsize: tuple = None, **kwargs):
    """
    Create a figure with academic styling applied.
    
    Returns
    -------
    fig, axes
    """
    set_academic_style()
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    return plt.subplots(n_rows, n_cols, figsize=figsize, **kwargs)
