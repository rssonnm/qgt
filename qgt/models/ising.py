"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.models.ising — Transverse-Field Ising Model
═══════════════════════════════════════════════════════════════════════════════

The canonical model for studying quantum phase transitions with QGT.

    H(J, h) = -J Σᵢ Z_i Z_{i+1} - h Σᵢ X_i

Phase transition at h/J = 1:
  - Ferromagnetic (h < J): symmetry-broken, ⟨Z⟩ ≠ 0
  - Paramagnetic (h > J): disordered, ⟨Z⟩ = 0

In the continuum limit → free Majorana fermion field theory.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional
import time

from qgt.models.base import QuantumModel
from qgt.core.operators import PauliOperators
from qgt.utils.linalg import embed_operator


class IsingModel(QuantumModel):
    """
    1D Transverse-Field Ising Model.
    
    Parameters
    ----------
    n_sites : int
        Number of lattice sites (qubits). Keep ≤ 12 for exact diag.
    periodic : bool
        Periodic boundary conditions.
    
    Examples
    --------
    >>> model = IsingModel(n_sites=4)
    >>> E, psi = model.ground_state(np.array([1.0, 0.5]))
    >>> gap = model.energy_gap(np.array([1.0, 1.0]))  # Should be small
    """
    
    def __init__(self, n_sites: int = 4, periodic: bool = True):
        super().__init__(
            name="Transverse-Field Ising Model",
            param_names=["J", "h"],
            n_sites=n_sites
        )
        self.periodic = periodic
        
        # Pre-compute operator terms
        self._pauli = PauliOperators()
        self._ZZ_terms = self._build_ZZ_terms()
        self._X_terms = self._build_X_terms()
    
    def _build_ZZ_terms(self):
        """Pre-compute all Z_i Z_{i+1} interaction terms."""
        terms = []
        n_bonds = self.n_sites if self.periodic else self.n_sites - 1
        for i in range(n_bonds):
            j = (i + 1) % self.n_sites
            ZZ = self._pauli.two_site_interaction(
                self._pauli.Z, self._pauli.Z, i, j, self.n_sites
            )
            terms.append(ZZ)
        return terms
    
    def _build_X_terms(self):
        """Pre-compute all X_i transverse field terms."""
        return [embed_operator(self._pauli.X_sp, i, self.n_sites)
                for i in range(self.n_sites)]
    
    def hamiltonian(self, params: np.ndarray) -> csr_matrix:
        """
        Build H(J, h) = -J Σ Z_i Z_{i+1} - h Σ X_i.
        
        Parameters
        ----------
        params : ndarray, [J, h]
        
        Returns
        -------
        csr_matrix
        """
        J, h = params[0], params[1]
        H = csr_matrix((self.dim, self.dim), dtype=float)
        for ZZ in self._ZZ_terms:
            H = H - J * ZZ
        for X in self._X_terms:
            H = H - h * X
        return H
    
    def ground_state(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute ground state via exact diagonalization (Lanczos).
        
        Parameters
        ----------
        params : ndarray, [J, h]
        
        Returns
        -------
        (energy, state_vector)
        """
        H = self.hamiltonian(params)
        E, V = eigsh(H, k=1, which='SA')
        return float(E[0]), V[:, 0]
    
    def order_parameter(self, params: np.ndarray) -> float:
        """
        Compute the order parameter ⟨Z⟩ (magnetization per site).
        
        In the ferromagnetic phase (h < J), this should be non-zero.
        In the paramagnetic phase (h > J), it should vanish.
        """
        M_z = self._pauli.total_magnetization('Z', self.n_sites)
        _, psi = self.ground_state(params)
        return float(abs(np.real(psi.conj() @ M_z @ psi))) / self.n_sites
    
    def correlation_function(self, params: np.ndarray,
                              site_i: int = 0,
                              site_j: Optional[int] = None) -> float:
        """
        Compute ⟨Z_i Z_j⟩ - ⟨Z_i⟩⟨Z_j⟩ (connected correlator).
        
        Parameters
        ----------
        params : ndarray, [J, h]
        site_i, site_j : int
            Site indices. Default j = N/2 (maximum separation for PBC).
        
        Returns
        -------
        float
            Connected correlation function.
        """
        if site_j is None:
            site_j = self.n_sites // 2
        
        _, psi = self.ground_state(params)
        
        Zi = embed_operator(self._pauli.Z_sp, site_i, self.n_sites)
        Zj = embed_operator(self._pauli.Z_sp, site_j, self.n_sites)
        ZiZj = Zi @ Zj
        
        exp_ZiZj = float(np.real(psi.conj() @ ZiZj @ psi))
        exp_Zi = float(np.real(psi.conj() @ Zi @ psi))
        exp_Zj = float(np.real(psi.conj() @ Zj @ psi))
        
        return exp_ZiZj - exp_Zi * exp_Zj
    
    @property
    def critical_point(self) -> dict:
        """Return the known critical point for the TFIM."""
        return {
            'condition': 'h/J = 1',
            'universality_class': '2D classical Ising',
            'critical_exponents': {
                'nu': 1.0,
                'z': 1.0,
                'eta': 0.25,
            },
            'description': (
                'The quantum phase transition occurs at h = J, '
                'separating the Ising-ordered (ferromagnetic) phase '
                'from the quantum disordered (paramagnetic) phase.'
            )
        }
