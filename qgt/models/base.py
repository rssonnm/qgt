"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.models.base — Abstract Base Class for Quantum Models
═══════════════════════════════════════════════════════════════════════════════

Defines the QuantumModel interface that all physical models must implement.
This ensures consistent interaction with the QGT, geometry, and
visualization modules.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class QuantumModel(ABC):
    """
    Abstract base class for parameterized quantum models.
    
    Every model must provide:
      1. A parameter space description
      2. A Hamiltonian constructor
      3. A ground state solver
    
    Subclasses
    ----------
    IsingModel : Transverse-field Ising model
    HeisenbergModel : XXZ Heisenberg chain
    SU2Model : SU(2) symmetric models
    
    Examples
    --------
    To implement a new model::
    
        class MyModel(QuantumModel):
            def __init__(self, n_sites):
                super().__init__(
                    name="My Model",
                    param_names=["J", "h"],
                    n_sites=n_sites
                )
            
            def hamiltonian(self, params):
                J, h = params
                # ... build H ...
                return H
            
            def ground_state(self, params):
                H = self.hamiltonian(params)
                # ... diagonalize ...
                return energy, state
    """
    
    def __init__(self, name: str, param_names: List[str],
                 n_sites: int, local_dim: int = 2):
        """
        Parameters
        ----------
        name : str
            Human-readable model name.
        param_names : list of str
            Names of the model parameters (e.g., ['J', 'h']).
        n_sites : int
            Number of lattice sites.
        local_dim : int
            Local Hilbert space dimension (2 for qubits).
        """
        self.name = name
        self.param_names = param_names
        self.n_params = len(param_names)
        self.n_sites = n_sites
        self.local_dim = local_dim
        self.dim = local_dim ** n_sites
    
    @abstractmethod
    def hamiltonian(self, params: np.ndarray):
        """
        Construct the Hamiltonian H(λ) for given parameters.
        
        Parameters
        ----------
        params : ndarray, shape (n_params,)
        
        Returns
        -------
        H : sparse matrix or ndarray
        """
        pass
    
    @abstractmethod
    def ground_state(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the ground state |ψ₀(λ)⟩.
        
        Parameters
        ----------
        params : ndarray, shape (n_params,)
        
        Returns
        -------
        energy : float
            Ground state energy.
        state : ndarray, shape (dim,)
            Ground state vector.
        """
        pass
    
    def energy_spectrum(self, params: np.ndarray,
                        k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the k lowest energy levels.
        
        Parameters
        ----------
        params : ndarray
        k : int
            Number of levels.
        
        Returns
        -------
        energies : ndarray, shape (k,)
        states : ndarray, shape (dim, k)
        """
        from scipy.sparse.linalg import eigsh
        H = self.hamiltonian(params)
        E, V = eigsh(H, k=k, which='SA')
        idx = np.argsort(E)
        return E[idx], V[:, idx]
    
    def energy_gap(self, params: np.ndarray) -> float:
        """
        Compute the energy gap Δ = E₁ - E₀.
        
        The gap closes at quantum phase transitions.
        """
        E, _ = self.energy_spectrum(params, k=2)
        return float(E[1] - E[0])
    
    def state_fn(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Wrapper for ground_state() compatible with QGT interface.
        
        Returns (energy, state) tuple.
        """
        return self.ground_state(params)
    
    def expectation_value(self, operator, params: np.ndarray) -> float:
        """
        Compute ⟨ψ₀|O|ψ₀⟩ for a given operator.
        
        Parameters
        ----------
        operator : sparse matrix or ndarray
        params : ndarray
        
        Returns
        -------
        float
        """
        _, psi = self.ground_state(params)
        if hasattr(operator, 'toarray'):
            return float(np.real(psi.conj() @ operator @ psi))
        else:
            return float(np.real(psi.conj() @ operator @ psi))
    
    def scan_energy_gap(self, param_ranges: list,
                         n_points: int = 30) -> dict:
        """
        Scan the energy gap across parameter space.
        
        Returns
        -------
        dict with 'param_0', 'param_1', 'gap' arrays
        """
        p0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points)
        p1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points)
        gap = np.zeros((n_points, n_points))
        
        for i, x in enumerate(p0):
            for j, y in enumerate(p1):
                try:
                    gap[i, j] = self.energy_gap(np.array([x, y]))
                except Exception:
                    gap[i, j] = np.nan
        
        return {'param_0': p0, 'param_1': p1, 'gap': gap}
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', n_sites={self.n_sites}, "
                f"dim={self.dim}, params={self.param_names})")
