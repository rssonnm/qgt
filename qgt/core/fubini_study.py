"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.core.fubini_study — Fubini-Study Metric
═══════════════════════════════════════════════════════════════════════════════

Dedicated module for the Fubini-Study metric tensor, providing:
  - Direct computation from quantum states
  - Relation to quantum Fisher information
  - Fidelity distance between states
  - Bures distance

The Fubini-Study metric is the natural Riemannian metric on the
projective Hilbert space CP^{n-1} (the space of pure quantum states).
"""

import numpy as np
from typing import Callable, Tuple

from qgt.core.qgt import QuantumGeometricTensor


class FubiniStudyMetric:
    """
    The Fubini-Study metric on quantum state space.
    
    For a parameterized family |ψ(θ)⟩, the Fubini-Study metric is:
        g_μν = Re[⟨∂_μψ|∂_νψ⟩ - ⟨∂_μψ|ψ⟩⟨ψ|∂_νψ⟩]
    
    This is the real part of the QGT and equals (1/4) × the quantum
    Fisher information matrix.
    
    Parameters
    ----------
    state_fn : callable
        Function mapping params → (energy, state_vector).
    n_params : int
        Number of parameters.
    epsilon : float
        Finite difference step size.
    
    Examples
    --------
    >>> fs = FubiniStudyMetric(my_state_fn, n_params=2)
    >>> g = fs.metric_tensor(np.array([1.0, 0.5]))
    >>> ds2 = fs.distance_squared(params_a, params_b)
    """
    
    def __init__(self, state_fn: Callable, n_params: int,
                 epsilon: float = 1e-4):
        self.state_fn = state_fn
        self.n_params = n_params
        self.eps = epsilon
        self._qgt = QuantumGeometricTensor(state_fn, n_params, epsilon)
    
    def metric_tensor(self, params: np.ndarray) -> np.ndarray:
        """
        Compute g_μν at the given parameters.
        
        Returns
        -------
        g : ndarray, shape (n_params, n_params)
            Symmetric, positive semi-definite metric tensor.
        """
        return self._qgt.metric(params)
    
    def inverse_metric(self, params: np.ndarray,
                       regularization: float = 1e-10) -> np.ndarray:
        """
        Compute g^{μν} (the inverse metric), with regularization.
        
        Used in the Quantum Natural Gradient:
            θ_new = θ - η · g^{-1} · ∇L
        
        Parameters
        ----------
        params : ndarray
        regularization : float
            Small value added to diagonal for numerical stability.
        
        Returns
        -------
        g_inv : ndarray, shape (n_params, n_params)
        """
        g = self.metric_tensor(params)
        g_reg = g + regularization * np.eye(self.n_params)
        return np.linalg.inv(g_reg)
    
    def distance_squared(self, params_a: np.ndarray,
                          params_b: np.ndarray) -> float:
        """
        Compute the infinitesimal Fubini-Study distance squared.
        
        ds² = Σ_μν g_μν δθ^μ δθ^ν
        
        For nearby points. For finite distances, use geodesic distance.
        """
        midpoint = (params_a + params_b) / 2
        delta = params_b - params_a
        g = self.metric_tensor(midpoint)
        return float(delta @ g @ delta)
    
    def quantum_fisher_information(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the Quantum Fisher Information matrix F = 4g.
        
        The QFI bounds the precision of parameter estimation
        via the quantum Cramér-Rao bound:
            Var(θ_μ) ≥ 1 / (N · F_μμ)
        """
        return 4.0 * self.metric_tensor(params)
    
    @staticmethod
    def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
        """Fidelity F = |⟨ψ|φ⟩|² between two pure states."""
        return float(abs(np.vdot(psi, phi)) ** 2)
    
    @staticmethod
    def bures_distance(psi: np.ndarray, phi: np.ndarray) -> float:
        """
        Bures distance D_B = √(2(1 - √F)) between pure states.
        
        This is the geodesic distance on the Bloch sphere for single qubits.
        """
        F = abs(np.vdot(psi, phi)) ** 2
        return float(np.sqrt(2 * (1 - np.sqrt(max(0, F)))))
    
    @staticmethod
    def fubini_study_distance(psi: np.ndarray, phi: np.ndarray) -> float:
        """
        Fubini-Study distance d_FS = arccos(√F).
        
        This is the angle between rays in projective Hilbert space.
        Range: [0, π/2].
        """
        F = abs(np.vdot(psi, phi)) ** 2
        return float(np.arccos(np.sqrt(np.clip(F, 0, 1))))
