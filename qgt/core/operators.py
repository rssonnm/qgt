"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.core.operators — Quantum Operators and Pauli Algebra
═══════════════════════════════════════════════════════════════════════════════

Provides Pauli matrices, common operators, and tools for constructing
many-body Hamiltonians from local interactions.
"""

import numpy as np
from scipy.sparse import csr_matrix, eye as sp_eye, kron as sp_kron
from typing import List, Optional

from qgt.utils.linalg import sparse_kron_chain, embed_operator


class PauliOperators:
    """
    Container for Pauli matrices and common quantum operators.
    
    Provides both dense (numpy) and sparse (scipy.sparse) versions.
    All operators are 2×2 by default; use embed() for many-body systems.
    
    Attributes
    ----------
    I : ndarray (2×2) — Identity
    X : ndarray (2×2) — Pauli-X (σₓ)
    Y : ndarray (2×2) — Pauli-Y (σᵧ)
    Z : ndarray (2×2) — Pauli-Z (σ_z)
    
    plus  : ndarray (2×2) — Raising operator σ⁺
    minus : ndarray (2×2) — Lowering operator σ⁻
    
    Examples
    --------
    >>> p = PauliOperators()
    >>> p.X @ p.Y  # = iZ
    array([[ 0.+1.j,  0.+0.j],
           [ 0.+0.j,  0.-1.j]])
    """
    
    def __init__(self):
        # Pauli matrices
        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Raising / Lowering
        self.plus = (self.X + 1j * self.Y) / 2   # σ⁺ = |0⟩⟨1|
        self.minus = (self.X - 1j * self.Y) / 2  # σ⁻ = |1⟩⟨0|
        
        # Sparse versions (for many-body)
        self.I_sp = csr_matrix(self.I)
        self.X_sp = csr_matrix(self.X)
        self.Y_sp = csr_matrix(self.Y)
        self.Z_sp = csr_matrix(self.Z)
        self.plus_sp = csr_matrix(self.plus)
        self.minus_sp = csr_matrix(self.minus)
    
    def n_body_operator(self, local_ops: dict, n_sites: int) -> csr_matrix:
        """
        Construct a many-body operator from local operators.
        
        Parameters
        ----------
        local_ops : dict
            Mapping site_index → operator_matrix.
            Sites not in dict get Identity.
        n_sites : int
            Total number of sites.
        
        Returns
        -------
        csr_matrix
            Full Hilbert space operator.
        
        Examples
        --------
        >>> p = PauliOperators()
        >>> ZZ_01 = p.n_body_operator({0: p.Z, 1: p.Z}, n_sites=4)
        """
        ops = [self.I_sp] * n_sites
        for site, op in local_ops.items():
            ops[site] = csr_matrix(op) if not isinstance(op, csr_matrix) else op
        return sparse_kron_chain(ops)
    
    def two_site_interaction(self, op_A: np.ndarray, op_B: np.ndarray,
                              site_i: int, site_j: int,
                              n_sites: int) -> csr_matrix:
        """
        Construct A_i ⊗ B_j two-site interaction operator.
        
        Parameters
        ----------
        op_A, op_B : ndarray
            Local operators.
        site_i, site_j : int
            Site indices.
        n_sites : int
        
        Returns
        -------
        csr_matrix
        """
        return self.n_body_operator({site_i: op_A, site_j: op_B}, n_sites)
    
    def total_magnetization(self, axis: str, n_sites: int) -> csr_matrix:
        """
        Compute total magnetization M = Σᵢ σᵢ along a given axis.
        
        Parameters
        ----------
        axis : str
            'X', 'Y', or 'Z'.
        n_sites : int
        
        Returns
        -------
        csr_matrix
        """
        op_map = {'X': self.X_sp, 'Y': self.Y_sp, 'Z': self.Z_sp}
        op = op_map[axis.upper()]
        
        M = csr_matrix((2**n_sites, 2**n_sites), dtype=complex)
        for site in range(n_sites):
            M = M + embed_operator(op, site, n_sites)
        return M


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: module-level Pauli matrices
# ═══════════════════════════════════════════════════════════════════════════

_pauli = PauliOperators()
sigma_x = _pauli.X
sigma_y = _pauli.Y
sigma_z = _pauli.Z
sigma_plus = _pauli.plus
sigma_minus = _pauli.minus
