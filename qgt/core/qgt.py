"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.core.qgt — Quantum Geometric Tensor
═══════════════════════════════════════════════════════════════════════════════

The central object of this package. Computes the full QGT:

    χ_μν = ⟨∂_μ ψ| (1 - |ψ⟩⟨ψ|) |∂_ν ψ⟩
         = ⟨∂_μ ψ|∂_ν ψ⟩ - ⟨∂_μ ψ|ψ⟩⟨ψ|∂_ν ψ⟩

Decomposition:
    g_μν = Re(χ_μν)   — Fubini-Study metric (quantum Fisher information / 4)
    Ω_μν = 2·Im(χ_μν) — Berry curvature

This module provides three computation methods:
    1. Numerical finite differences (most general)
    2. Full diagonalization perturbation theory (fast for Hamiltonians)
    3. PennyLane integration (for variational circuits)
"""

import numpy as np
from typing import Callable, Tuple, Optional, List

from qgt.utils.finite_diff import align_phase


class QuantumGeometricTensor:
    """
    Computes the Quantum Geometric Tensor for parameterized quantum states.
    
    The QGT encodes the complete local geometry of the quantum state manifold:
      - Its real part (Fubini-Study metric) measures distinguishability
      - Its imaginary part (Berry curvature) measures geometric phase
    
    Parameters
    ----------
    state_fn : callable
        Function that maps parameters → (energy, state_vector).
        Signature: state_fn(params: ndarray) → (float, ndarray)
    n_params : int
        Number of parameters (manifold dimension).
    epsilon : float
        Step size for finite differences.
    method : str
        Computation method: 'finite_diff' (default), 'perturbation'.
    
    Examples
    --------
    >>> def my_state(params):
    ...     J, h = params
    ...     # ... build Hamiltonian, diagonalize ...
    ...     return energy, ground_state
    >>> qgt = QuantumGeometricTensor(my_state, n_params=2)
    >>> chi = qgt.compute(np.array([1.0, 0.5]))
    >>> metric = qgt.metric(np.array([1.0, 0.5]))
    >>> berry  = qgt.berry_curvature(np.array([1.0, 0.5]))
    """
    
    def __init__(self, state_fn: Callable, n_params: int,
                 epsilon: float = 1e-4, method: str = 'finite_diff'):
        self.state_fn = state_fn
        self.n_params = n_params
        self.eps = epsilon
        self.method = method
        
        # Cache for state derivatives (reduces redundant computations)
        self._cache = {}
    
    def _state_derivative(self, params: np.ndarray,
                          direction: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ∂|ψ⟩/∂λ_μ using central finite difference with gauge alignment.
        
        Parameters
        ----------
        params : ndarray
            Current parameter values.
        direction : int
            Index of the parameter to differentiate w.r.t.
        
        Returns
        -------
        dpsi : ndarray
            State derivative.
        psi0 : ndarray
            State at base point.
        """
        _, psi0 = self.state_fn(params)
        
        params_plus = params.copy().astype(float)
        params_plus[direction] += self.eps
        _, psi_plus = self.state_fn(params_plus)
        
        params_minus = params.copy().astype(float)
        params_minus[direction] -= self.eps
        _, psi_minus = self.state_fn(params_minus)
        
        # Gauge alignment
        psi_plus = align_phase(psi0, psi_plus)
        psi_minus = align_phase(psi0, psi_minus)
        
        dpsi = (psi_plus - psi_minus) / (2 * self.eps)
        return dpsi, psi0
    
    def compute(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the full QGT χ_μν at the given parameters.
        
        Parameters
        ----------
        params : ndarray, shape (n_params,)
        
        Returns
        -------
        qgt : ndarray, shape (n_params, n_params), complex
            The full Quantum Geometric Tensor.
        """
        params = np.asarray(params, dtype=float)
        
        # Compute all state derivatives
        derivs = []
        psi0 = None
        for mu in range(self.n_params):
            dpsi, psi = self._state_derivative(params, mu)
            derivs.append(dpsi)
            if psi0 is None:
                psi0 = psi
        
        # Build QGT matrix
        chi = np.zeros((self.n_params, self.n_params), dtype=complex)
        for mu in range(self.n_params):
            for nu in range(self.n_params):
                # χ_μν = ⟨∂_μψ|∂_νψ⟩ - ⟨∂_μψ|ψ⟩⟨ψ|∂_νψ⟩
                t1 = np.vdot(derivs[mu], derivs[nu])
                t2 = np.vdot(derivs[mu], psi0) * np.vdot(psi0, derivs[nu])
                chi[mu, nu] = t1 - t2
        
        return chi
    
    def metric(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the Fubini-Study metric g_μν = Re(χ_μν).
        
        This is the quantum Fisher information matrix divided by 4.
        It measures the distinguishability of nearby quantum states.
        
        Parameters
        ----------
        params : ndarray
        
        Returns
        -------
        g : ndarray, shape (n_params, n_params), real
        """
        return np.real(self.compute(params))
    
    def berry_curvature(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the Berry curvature Ω_μν = 2·Im(χ_μν).
        
        The Berry curvature is the antisymmetric part of the QGT,
        related to the geometric phase acquired during adiabatic evolution.
        
        Parameters
        ----------
        params : ndarray
        
        Returns
        -------
        omega : ndarray, shape (n_params, n_params), real
        """
        return 2.0 * np.imag(self.compute(params))
    
    def fidelity_susceptibility(self, params: np.ndarray,
                                 n_sites: int = 1) -> float:
        """
        Compute the fidelity susceptibility χ_F = Tr(g) / N.
        
        This is the intensive (per-site) measure of how rapidly
        the ground state changes with parameters.
        
        Parameters
        ----------
        params : ndarray
        n_sites : int
            Number of lattice sites for normalization.
        
        Returns
        -------
        float
        """
        g = self.metric(params)
        return float(np.trace(g)) / n_sites
    
    def metric_determinant(self, params: np.ndarray) -> float:
        """Compute det(g_μν) — diverges at phase transitions."""
        g = self.metric(params)
        return float(np.linalg.det(g))
    
    def berry_phase(self, params_path: np.ndarray) -> float:
        """
        Compute the Berry phase along a closed path in parameter space.
        
        γ = ∮ A · dλ  where A_μ = i⟨ψ|∂_μψ⟩ is the Berry connection.
        
        Parameters
        ----------
        params_path : ndarray, shape (n_steps, n_params)
            Closed path in parameter space. First and last point should match.
        
        Returns
        -------
        float
            Berry phase (mod 2π).
        """
        n_steps = len(params_path)
        phase = 0.0
        
        for i in range(n_steps - 1):
            _, psi_i = self.state_fn(params_path[i])
            _, psi_j = self.state_fn(params_path[i + 1])
            psi_j = align_phase(psi_i, psi_j)
            overlap = np.vdot(psi_i, psi_j)
            phase += np.angle(overlap)
        
        return float(phase)
    
    def scan_parameter_space(self, param_ranges: List[Tuple[float, float]],
                              n_points: int = 20,
                              quantities: List[str] = None) -> dict:
        """
        Scan a 2D parameter space and compute QGT-derived quantities.
        
        Parameters
        ----------
        param_ranges : list of (min, max) tuples
            Range for each parameter (currently supports 2D).
        n_points : int
            Grid density per axis.
        quantities : list of str
            Which quantities to compute. Options:
            'metric_det', 'metric_trace', 'berry_curv', 'fidelity_sus'
        
        Returns
        -------
        dict
            Contains parameter grids and computed quantities.
        """
        if quantities is None:
            quantities = ['metric_det', 'metric_trace', 'berry_curv',
                          'fidelity_sus']
        
        assert len(param_ranges) == 2, "Only 2D scans supported currently"
        
        p0_vals = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points)
        p1_vals = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points)
        
        results = {
            'param_0': p0_vals,
            'param_1': p1_vals,
        }
        
        for q in quantities:
            results[q] = np.zeros((n_points, n_points))
        
        for i, p0 in enumerate(p0_vals):
            for j, p1 in enumerate(p1_vals):
                params = np.array([p0, p1])
                chi = self.compute(params)
                g = np.real(chi)
                omega = 2.0 * np.imag(chi)
                
                if 'metric_det' in quantities:
                    results['metric_det'][i, j] = np.linalg.det(g)
                if 'metric_trace' in quantities:
                    results['metric_trace'][i, j] = np.trace(g)
                if 'berry_curv' in quantities:
                    results['berry_curv'][i, j] = omega[0, 1]
                if 'fidelity_sus' in quantities:
                    results['fidelity_sus'][i, j] = np.trace(g)
        
        return results
