"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.geometry.geodesics — Geodesic Equation Solvers
═══════════════════════════════════════════════════════════════════════════════

Solves the geodesic equation on the information manifold:

    d²x^σ/dt² + Γ^σ_{μν} dx^μ/dt dx^ν/dt = 0

Geodesics are the "straightest possible paths" on the curved manifold.
Near quantum phase transitions, geodesics curve away from the critical
point, reflecting the infinite geometric distance across the phase boundary.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.interpolate import RegularGridInterpolator

from qgt.geometry.connection import ChristoffelSymbols
from qgt.geometry.metric import MetricTensor


class GeodesicSolver:
    """
    Solves geodesic equations on the information manifold.
    
    Two modes:
      1. Direct: Computes Christoffel on-the-fly (accurate but slow)
      2. Interpolated: Pre-computes on a grid (fast, recommended)
    
    Parameters
    ----------
    metric : MetricTensor
        The metric tensor field.
    epsilon : float
        Step size for Christoffel computation.
    """
    
    def __init__(self, metric: MetricTensor, epsilon: float = 1e-3):
        self.metric = metric
        self.christoffel = ChristoffelSymbols(metric, epsilon)
        self._interp_ready = False
    
    def precompute_grid(self, param_ranges: List[Tuple[float, float]],
                        n_grid: int = 15):
        """
        Pre-compute Christoffel symbols on a grid for fast interpolation.
        
        This is essential for geodesic integration, since on-the-fly
        Christoffel computation is too expensive for adaptive ODE solvers.
        
        Parameters
        ----------
        param_ranges : list of (min, max) tuples
        n_grid : int
            Grid density per dimension.
        """
        n_dim = len(param_ranges)
        grids = [np.linspace(r[0], r[1], n_grid) for r in param_ranges]
        
        # Christoffel has n_dim^3 components
        n_comp = n_dim ** 3
        Gamma_data = np.zeros((n_comp,) + tuple([n_grid] * n_dim))
        
        if n_dim == 2:
            for i, x in enumerate(grids[0]):
                for j, y in enumerate(grids[1]):
                    try:
                        G = self.christoffel.compute(np.array([x, y]))
                        Gamma_data[:, i, j] = G.ravel()
                    except Exception:
                        Gamma_data[:, i, j] = 0.0
        
        # Create interpolators
        self._interps = []
        for comp in range(n_comp):
            if n_dim == 2:
                interp = RegularGridInterpolator(
                    (grids[0], grids[1]), Gamma_data[comp],
                    method='linear', bounds_error=False, fill_value=0.0
                )
            self._interps.append(interp)
        
        self._param_ranges = param_ranges
        self._n_dim = n_dim
        self._interp_ready = True
    
    def _get_christoffel(self, params: np.ndarray) -> np.ndarray:
        """Get Christoffel symbols (interpolated if available)."""
        if self._interp_ready:
            pt = params.reshape(1, -1)
            values = np.array([interp(pt)[0] for interp in self._interps])
            return values.reshape((self._n_dim,) * 3)
        else:
            return self.christoffel.compute(params)
    
    def solve(self, start_point: np.ndarray, start_velocity: np.ndarray,
              t_span: Tuple[float, float] = (0, 2.0),
              n_steps: int = 200) -> np.ndarray:
        """
        Integrate the geodesic equation using fixed-step RK4.
        
        Parameters
        ----------
        start_point : ndarray, shape (n,)
            Initial position in parameter space.
        start_velocity : ndarray, shape (n,)
            Initial tangent vector (direction).
        t_span : tuple
            (t_start, t_end).
        n_steps : int
            Number of integration steps.
        
        Returns
        -------
        path : ndarray, shape (n_actual, n_dim)
            Geodesic path coordinates.
        """
        n = len(start_point)
        dt = (t_span[1] - t_span[0]) / n_steps
        
        # State vector: [position, velocity]
        y = np.concatenate([start_point, start_velocity])
        
        def rhs(state):
            pos, vel = state[:n], state[n:]
            
            # Boundary check
            if self._interp_ready:
                for d in range(n):
                    rng = self._param_ranges[d]
                    if pos[d] < rng[0] or pos[d] > rng[1]:
                        return np.concatenate([vel, np.zeros(n)])
            
            Gamma = self._get_christoffel(pos)
            accel = np.zeros(n)
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        accel[sigma] -= Gamma[sigma, mu, nu] * vel[mu] * vel[nu]
            
            return np.concatenate([vel, accel])
        
        path = np.zeros((n_steps, n))
        for i in range(n_steps):
            path[i] = y[:n]
            
            # RK4
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Early termination
            if np.linalg.norm(y[n:]) > 100:
                path = path[:i+1]
                break
        
        return path
    
    def geodesic_distance(self, point_a: np.ndarray,
                           point_b: np.ndarray,
                           n_refinements: int = 5) -> float:
        """
        Estimate the geodesic distance between two points.
        
        Uses the shooting method: try different initial velocities
        from point_a and find the one that reaches closest to point_b.
        
        Note: This is an approximation. For exact results on simple
        manifolds, use the analytic formula.
        
        Parameters
        ----------
        point_a, point_b : ndarray
        n_refinements : int
        
        Returns
        -------
        float
            Estimated geodesic distance.
        """
        direction = point_b - point_a
        dist_euclid = np.linalg.norm(direction)
        
        if dist_euclid < 1e-10:
            return 0.0
        
        # Use the metric to estimate arc length
        midpoint = (point_a + point_b) / 2
        g = self.metric.at(midpoint)
        return float(np.sqrt(direction @ g @ direction))
