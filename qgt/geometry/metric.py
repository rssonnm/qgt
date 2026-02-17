"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.geometry.metric — Metric Tensor Operations
═══════════════════════════════════════════════════════════════════════════════

Higher-level operations on the Fubini-Study metric tensor including
determinant, eigenvalues, condition number, and volume element.
"""

import numpy as np
from typing import Callable


class MetricTensor:
    """
    Operations on a 2D metric tensor field g_μν(x).
    
    Given a function that computes g at any point, this class provides
    geometric quantities derived from the metric.
    
    Parameters
    ----------
    metric_fn : callable(params) → ndarray (2×2)
        Function returning the metric tensor at given parameters.
    """
    
    def __init__(self, metric_fn: Callable):
        self.metric_fn = metric_fn
    
    def at(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the metric at given parameters."""
        return self.metric_fn(params)
    
    def inverse(self, params: np.ndarray,
                regularization: float = 1e-12) -> np.ndarray:
        """Compute g^{μν} with optional regularization."""
        g = self.at(params)
        return np.linalg.inv(g + regularization * np.eye(g.shape[0]))
    
    def determinant(self, params: np.ndarray) -> float:
        """Compute det(g) — the volume element √g for integration."""
        return float(np.linalg.det(self.at(params)))
    
    def trace(self, params: np.ndarray) -> float:
        """Compute Tr(g) — related to fidelity susceptibility."""
        return float(np.trace(self.at(params)))
    
    def eigenvalues(self, params: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of g — principal stretching factors."""
        return np.linalg.eigvalsh(self.at(params))
    
    def condition_number(self, params: np.ndarray) -> float:
        """
        Condition number κ(g) = λ_max / λ_min.
        
        Large κ indicates anisotropic stretching (common near phase transitions).
        """
        evals = self.eigenvalues(params)
        evals = evals[evals > 1e-15]
        if len(evals) < 2:
            return float('inf')
        return float(evals.max() / evals.min())
    
    def volume_element(self, params: np.ndarray) -> float:
        """Compute √(det g) — the natural volume form."""
        det = self.determinant(params)
        return float(np.sqrt(max(0, det)))
    
    def integrate_volume(self, param_ranges: list,
                          n_points: int = 30) -> float:
        """
        Numerically integrate the volume ∫ √g d²λ over a rectangular region.
        
        Parameters
        ----------
        param_ranges : list of (min, max) tuples
        n_points : int
        
        Returns
        -------
        float
            Total manifold volume.
        """
        p0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points)
        p1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points)
        dp0 = p0[1] - p0[0]
        dp1 = p1[1] - p1[0]
        
        vol = 0.0
        for x in p0:
            for y in p1:
                vol += self.volume_element(np.array([x, y])) * dp0 * dp1
        return vol
