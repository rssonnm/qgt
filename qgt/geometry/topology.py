"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.geometry.topology — Topological Invariants
═══════════════════════════════════════════════════════════════════════════════

Computes topological quantities from the QGT:
  - Berry phase (loop integral of Berry connection)
  - Chern number (integral of Berry curvature over closed surface)
  - Winding number
  - Euler characteristic

These quantities are quantized (integers or fractions thereof)
and robust against continuous deformations — they capture the
global structure of the quantum state manifold.
"""

import numpy as np
from typing import Callable, Tuple


class TopologicalInvariants:
    """
    Computes topological invariants from the Berry curvature.
    
    Parameters
    ----------
    berry_curvature_fn : callable(params) → ndarray
        Function returning Berry curvature Ω_{μν} at given parameters.
    n_params : int
        Dimension of parameter space.
    """
    
    def __init__(self, berry_curvature_fn: Callable, n_params: int = 2):
        self.berry_fn = berry_curvature_fn
        self.n_params = n_params
    
    def chern_number(self, param_ranges: list,
                      n_points: int = 50) -> float:
        """
        Compute the first Chern number over a 2D parameter surface.
        
        C₁ = (1/2π) ∫∫ Ω_{12} dp₁ dp₂
        
        For a non-degenerate ground state of a gapped system,
        this should be an integer (topological quantization).
        
        Parameters
        ----------
        param_ranges : list of (min, max) tuples
            Integration region.
        n_points : int
            Grid density for numerical integration.
        
        Returns
        -------
        float
            First Chern number (should be close to an integer).
        """
        assert len(param_ranges) == 2, "Chern number requires 2D parameter space"
        
        p0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points)
        p1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points)
        dp0 = p0[1] - p0[0]
        dp1 = p1[1] - p1[0]
        
        integral = 0.0
        for x in p0:
            for y in p1:
                omega = self.berry_fn(np.array([x, y]))
                if omega.ndim == 2:
                    integral += omega[0, 1] * dp0 * dp1
                else:
                    integral += omega * dp0 * dp1
        
        return integral / (2 * np.pi)
    
    def berry_phase_loop(self, state_fn: Callable,
                          path: np.ndarray) -> float:
        """
        Compute the Berry phase along a discrete closed path.
        
        γ = -Im Σᵢ log⟨ψᵢ|ψᵢ₊₁⟩
        
        Uses the discretized Wilson loop formula for numerical stability.
        
        Parameters
        ----------
        state_fn : callable(params) → (energy, state)
        path : ndarray, shape (n_points, n_params)
            Closed path (first and last points should overlap or be close).
        
        Returns
        -------
        float
            Berry phase (mod 2π).
        """
        n_pts = len(path)
        phase = 0.0
        
        for i in range(n_pts):
            _, psi_i = state_fn(path[i])
            _, psi_j = state_fn(path[(i + 1) % n_pts])
            
            overlap = np.vdot(psi_i, psi_j)
            phase += np.angle(overlap)
        
        return float(phase)
    
    def winding_number(self, state_fn: Callable,
                        center: np.ndarray,
                        radius: float = 0.5,
                        n_points: int = 100) -> float:
        """
        Compute the winding number by integrating the Berry phase
        around a circular loop centered at a given point.
        
        Parameters
        ----------
        state_fn : callable(params) → (energy, state)
        center : ndarray, shape (2,)
        radius : float
        n_points : int
        
        Returns
        -------
        float
            Winding number (should be close to an integer).
        """
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        path = np.array([
            center + radius * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ])
        
        phase = self.berry_phase_loop(state_fn, path)
        return phase / (2 * np.pi)
    
    @staticmethod
    def euler_characteristic_2d(scalar_curvature_fn: Callable,
                                 volume_element_fn: Callable,
                                 param_ranges: list,
                                 n_points: int = 30) -> float:
        """
        Compute the Euler characteristic via the Gauss-Bonnet theorem.
        
        χ = (1/4π) ∫ R √g d²x
        
        For a closed 2D manifold, χ = 2(1-g) where g is the genus.
        
        Parameters
        ----------
        scalar_curvature_fn : callable(params) → float
        volume_element_fn : callable(params) → float
        param_ranges : list of (min, max)
        n_points : int
        
        Returns
        -------
        float
        """
        p0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points)
        p1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points)
        dp0 = p0[1] - p0[0]
        dp1 = p1[1] - p1[0]
        
        integral = 0.0
        for x in p0:
            for y in p1:
                params = np.array([x, y])
                try:
                    R = scalar_curvature_fn(params)
                    sqrt_g = volume_element_fn(params)
                    integral += R * sqrt_g * dp0 * dp1
                except Exception:
                    pass
        
        return integral / (4 * np.pi)
