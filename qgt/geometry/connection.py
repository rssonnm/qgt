"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.geometry.connection — Christoffel Symbols and Connections
═══════════════════════════════════════════════════════════════════════════════

Computes the Levi-Civita connection (Christoffel symbols) from the
Fubini-Study metric tensor on the information manifold.

    Γ^σ_μν = ½ g^{σρ} (∂_μ g_{ρν} + ∂_ν g_{ρμ} - ∂_ρ g_{μν})
"""

import numpy as np
from typing import Callable

from qgt.geometry.metric import MetricTensor


class ChristoffelSymbols:
    """
    Computes Christoffel symbols of the Levi-Civita connection.
    
    Parameters
    ----------
    metric : MetricTensor
        The metric tensor field.
    epsilon : float
        Step size for finite-difference metric derivatives.
    """
    
    def __init__(self, metric: MetricTensor, epsilon: float = 1e-3):
        self.metric = metric
        self.eps = epsilon
    
    def compute(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Γ^σ_{μν} at the given point.
        
        Parameters
        ----------
        params : ndarray, shape (n,)
        
        Returns
        -------
        Gamma : ndarray, shape (n, n, n)
            Gamma[sigma, mu, nu] = Γ^σ_{μν}
        """
        n = len(params)
        g = self.metric.at(params)
        g_inv = np.linalg.inv(g)
        
        # Compute metric derivatives: dg[rho, mu, nu] = ∂_rho g_{mu,nu}
        dg = np.zeros((n, n, n))
        for rho in range(n):
            e_rho = np.zeros(n)
            e_rho[rho] = 1.0
            g_plus = self.metric.at(params + self.eps * e_rho)
            g_minus = self.metric.at(params - self.eps * e_rho)
            dg[rho] = (g_plus - g_minus) / (2 * self.eps)
        
        # Christoffel: Γ^σ_μν = ½ g^{σρ} (∂_μ g_{ρν} + ∂_ν g_{ρμ} - ∂_ρ g_{μν})
        Gamma = np.zeros((n, n, n))
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    for rho in range(n):
                        Gamma[sigma, mu, nu] += 0.5 * g_inv[sigma, rho] * (
                            dg[mu, rho, nu] + dg[nu, rho, mu] - dg[rho, mu, nu]
                        )
        return Gamma
    
    def parallel_transport(self, vector: np.ndarray, params: np.ndarray,
                            direction: np.ndarray,
                            dt: float = 0.01) -> np.ndarray:
        """
        Infinitesimally parallel-transport a vector along a direction.
        
        dV^σ/dt = -Γ^σ_{μν} V^μ ẋ^ν
        
        Parameters
        ----------
        vector : ndarray, shape (n,)
            Vector to transport.
        params : ndarray, shape (n,)
            Current point.
        direction : ndarray, shape (n,)
            Direction of transport (tangent vector ẋ).
        dt : float
            Infinitesimal step.
        
        Returns
        -------
        ndarray
            Transported vector.
        """
        Gamma = self.compute(params)
        n = len(params)
        
        dV = np.zeros(n)
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    dV[sigma] -= Gamma[sigma, mu, nu] * vector[mu] * direction[nu]
        
        return vector + dV * dt
