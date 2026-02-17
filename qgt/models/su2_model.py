"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.models.su2_model — SU(2)-Symmetric Quantum Models
═══════════════════════════════════════════════════════════════════════════════

A general SU(2)-symmetric spin model with configurable interactions.
Useful for studying the role of symmetry in the QGT structure.

    H(J₁, J₂) = J₁ Σ S_i · S_{i+1}  +  J₂ Σ S_i · S_{i+2}
                 (nearest-neighbor)       (next-nearest-neighbor)

When J₂/J₁ exceeds a critical value (~0.2412 for the 1D chain),
the system transitions from a Néel-ordered phase to a dimerized phase
(Majumdar-Ghosh point at J₂/J₁ = 0.5).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple

from qgt.models.base import QuantumModel
from qgt.core.operators import PauliOperators
from qgt.utils.linalg import embed_operator


class SU2Model(QuantumModel):
    """
    J₁-J₂ Heisenberg model with SU(2) symmetry.
    
    H = J₁ Σ_{⟨i,j⟩} S_i · S_j + J₂ Σ_{⟨⟨i,j⟩⟩} S_i · S_j
    
    where S_i · S_j = ½(X_i X_j + Y_i Y_j + Z_i Z_j).
    
    Parameters
    ----------
    n_sites : int
    periodic : bool
    
    Examples
    --------
    >>> model = SU2Model(n_sites=4)
    >>> # Majumdar-Ghosh point: J2/J1 = 0.5
    >>> E, psi = model.ground_state(np.array([1.0, 0.5]))
    """
    
    def __init__(self, n_sites: int = 4, periodic: bool = True):
        super().__init__(
            name="J1-J2 Heisenberg (SU(2))",
            param_names=["J1", "J2"],
            n_sites=n_sites
        )
        self.periodic = periodic
        
        self._pauli = PauliOperators()
        self._nn_terms = self._build_heisenberg_terms(1)   # nearest-neighbor
        self._nnn_terms = self._build_heisenberg_terms(2)  # next-nearest-neighbor
    
    def _build_heisenberg_terms(self, distance: int):
        """Build S_i · S_j terms at a given distance."""
        terms = []
        n_bonds = self.n_sites if self.periodic else self.n_sites - distance
        
        for i in range(n_bonds):
            j = (i + distance) % self.n_sites
            SdotS = csr_matrix((self.dim, self.dim), dtype=complex)
            for axis in ['X', 'Y', 'Z']:
                op = {'X': self._pauli.X, 'Y': self._pauli.Y,
                      'Z': self._pauli.Z}[axis]
                SdotS = SdotS + 0.25 * self._pauli.two_site_interaction(
                    op, op, i, j, self.n_sites
                )
            terms.append(SdotS)
        return terms
    
    def hamiltonian(self, params: np.ndarray) -> csr_matrix:
        """Build H = J₁ Σ S·S_nn + J₂ Σ S·S_nnn."""
        J1, J2 = params[0], params[1]
        H = csr_matrix((self.dim, self.dim), dtype=complex)
        
        for term in self._nn_terms:
            H = H + J1 * term
        for term in self._nnn_terms:
            H = H + J2 * term
        
        return H
    
    def ground_state(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute ground state via Lanczos."""
        H = self.hamiltonian(params)
        E, V = eigsh(H, k=1, which='SA')
        return float(E[0]), V[:, 0]
    
    def dimer_order_parameter(self, params: np.ndarray) -> float:
        """
        Compute the dimer order parameter:
            D = (1/N) Σᵢ (-1)ⁱ ⟨S_i · S_{i+1}⟩
        
        Non-zero in the dimerized phase (J₂/J₁ > 0.2412).
        """
        _, psi = self.ground_state(params)
        D = 0.0
        
        for idx, term in enumerate(self._nn_terms):
            sign = (-1) ** idx
            D += sign * float(np.real(psi.conj() @ term @ psi))
        
        return abs(D) / self.n_sites
    
    @property
    def critical_point(self) -> dict:
        return {
            'condition': 'J2/J1 ≈ 0.2412',
            'type': 'Néel → Dimer (deconfined quantum critical point)',
            'special_points': {
                'J2/J1 = 0': 'Heisenberg antiferromagnet',
                'J2/J1 = 0.5': 'Majumdar-Ghosh (exact dimer ground state)',
            }
        }
