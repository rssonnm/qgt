"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.geometry.curvature — Curvature Tensors
═══════════════════════════════════════════════════════════════════════════════

Computes the full curvature hierarchy:
    Riemann tensor  → Ricci tensor → Scalar curvature

For 2D manifolds: R = 2K (Gaussian curvature).
"""

import numpy as np
from typing import Callable

from qgt.geometry.metric import MetricTensor
from qgt.geometry.connection import ChristoffelSymbols


class CurvatureAnalyzer:
    """
    Computes curvature quantities on the information manifold.
    
    Parameters
    ----------
    metric : MetricTensor
    epsilon : float
        Step size for numerical derivatives.
    """
    
    def __init__(self, metric: MetricTensor, epsilon: float = 1e-3):
        self.metric = metric
        self.eps = epsilon
        self.christoffel = ChristoffelSymbols(metric, epsilon)
    
    def riemann_tensor(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the Riemann curvature tensor R^σ_{ρμν}.
        
        R^σ_{ρμν} = ∂_μ Γ^σ_{νρ} - ∂_ν Γ^σ_{μρ}
                   + Γ^σ_{μλ}Γ^λ_{νρ} - Γ^σ_{νλ}Γ^λ_{μρ}
        
        Parameters
        ----------
        params : ndarray, shape (n,)
        
        Returns
        -------
        R : ndarray, shape (n, n, n, n)
            Riemann[sigma, rho, mu, nu]
        """
        n = len(params)
        Gamma_c = self.christoffel.compute(params)
        
        # Compute Christoffel derivatives via finite differences
        dGamma = np.zeros((n,) + Gamma_c.shape)
        for d in range(n):
            e_d = np.zeros(n)
            e_d[d] = 1.0
            G_plus = self.christoffel.compute(params + self.eps * e_d)
            G_minus = self.christoffel.compute(params - self.eps * e_d)
            dGamma[d] = (G_plus - G_minus) / (2 * self.eps)
        
        # Riemann tensor
        R = np.zeros((n, n, n, n))
        for s in range(n):
            for rho in range(n):
                for mu in range(n):
                    for nu in range(n):
                        R[s, rho, mu, nu] = (
                            dGamma[mu, s, nu, rho] - dGamma[nu, s, mu, rho]
                        )
                        for lam in range(n):
                            R[s, rho, mu, nu] += (
                                Gamma_c[s, mu, lam] * Gamma_c[lam, nu, rho]
                              - Gamma_c[s, nu, lam] * Gamma_c[lam, mu, rho]
                            )
        return R
    
    def ricci_tensor(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the Ricci tensor R_{ρν} = R^μ_{ρμν}.
        
        Parameters
        ----------
        params : ndarray, shape (n,)
        
        Returns
        -------
        Ricci : ndarray, shape (n, n)
        """
        R = self.riemann_tensor(params)
        n = len(params)
        Ricci = np.zeros((n, n))
        for rho in range(n):
            for nu in range(n):
                for mu in range(n):
                    Ricci[rho, nu] += R[mu, rho, mu, nu]
        return Ricci
    
    def scalar_curvature(self, params: np.ndarray) -> float:
        """
        Compute the Ricci scalar R = g^{ρν} R_{ρν}.
        
        For a 2D manifold, R = 2K where K is the Gaussian curvature.
        
        Physical significance: R diverges at quantum phase transitions,
        providing a coordinate-independent signature of criticality.
        
        Parameters
        ----------
        params : ndarray, shape (n,)
        
        Returns
        -------
        float
            Scalar curvature.
        """
        g = self.metric.at(params)
        g_inv = np.linalg.inv(g)
        Ricci = self.ricci_tensor(params)
        return float(np.einsum('ij,ij', g_inv, Ricci))
    
    def gaussian_curvature(self, params: np.ndarray) -> float:
        """
        Compute the Gaussian curvature K = R/2 (valid for 2D only).
        
        Returns
        -------
        float
        """
        return self.scalar_curvature(params) / 2.0
    
    def scan_curvature(self, param_ranges: list,
                        n_points: int = 15) -> dict:
        """
        Scan scalar curvature over a 2D parameter grid.
        
        Parameters
        ----------
        param_ranges : list of (min, max)
        n_points : int
        
        Returns
        -------
        dict with 'param_0', 'param_1', 'scalar_curvature' arrays
        """
        p0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points)
        p1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points)
        R_grid = np.zeros((n_points, n_points))
        
        for i, x in enumerate(p0):
            for j, y in enumerate(p1):
                try:
                    R_grid[i, j] = self.scalar_curvature(np.array([x, y]))
                except Exception:
                    R_grid[i, j] = np.nan
        
        return {'param_0': p0, 'param_1': p1, 'scalar_curvature': R_grid}
