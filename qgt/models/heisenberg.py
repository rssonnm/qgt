"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.models.heisenberg — XXZ Heisenberg Chain
═══════════════════════════════════════════════════════════════════════════════

    H(J, Δ, h) = J Σᵢ [X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1}]
                 - h Σᵢ Z_i

Phase diagram (for J > 0):
  - Δ < -1: Ferromagnetic (↑↑↑...)
  - -1 < Δ < 1: XY / Luttinger liquid (critical, gapless)
  - Δ > 1: Antiferromagnetic Néel (↑↓↑↓...)

The QGT reveals different geometric structures in each phase.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple

from qgt.models.base import QuantumModel
from qgt.core.operators import PauliOperators
from qgt.utils.linalg import embed_operator


class HeisenbergModel(QuantumModel):
    """
    1D XXZ Heisenberg Model.
    
    Parameters
    ----------
    n_sites : int
        Number of sites.
    periodic : bool
        Periodic boundary conditions.
    
    Examples
    --------
    >>> model = HeisenbergModel(n_sites=4)
    >>> # Isotropic point (XXX model): Δ = 1
    >>> E, psi = model.ground_state(np.array([1.0, 1.0, 0.0]))
    """
    
    def __init__(self, n_sites: int = 4, periodic: bool = True):
        super().__init__(
            name="XXZ Heisenberg Chain",
            param_names=["J", "Delta", "h"],
            n_sites=n_sites
        )
        self.periodic = periodic
        
        self._pauli = PauliOperators()
        self._XX_terms = self._build_interaction('X')
        self._YY_terms = self._build_interaction('Y')
        self._ZZ_terms = self._build_interaction('Z')
        self._Z_terms = [embed_operator(self._pauli.Z_sp, i, self.n_sites)
                          for i in range(self.n_sites)]
    
    def _build_interaction(self, axis: str):
        """Build nearest-neighbor interaction terms."""
        op = {'X': self._pauli.X, 'Y': self._pauli.Y, 'Z': self._pauli.Z}[axis]
        terms = []
        n_bonds = self.n_sites if self.periodic else self.n_sites - 1
        for i in range(n_bonds):
            j = (i + 1) % self.n_sites
            terms.append(self._pauli.two_site_interaction(
                op, op, i, j, self.n_sites
            ))
        return terms
    
    def hamiltonian(self, params: np.ndarray) -> csr_matrix:
        """
        Build H = J[Σ(XX + YY + ΔZZ)] - hΣZ.
        
        Parameters
        ----------
        params : ndarray, [J, Δ, h]
        """
        J, Delta, h = params[0], params[1], params[2]
        H = csr_matrix((self.dim, self.dim), dtype=complex)
        
        for XX in self._XX_terms:
            H = H + J * XX
        for YY in self._YY_terms:
            H = H + J * YY
        for ZZ in self._ZZ_terms:
            H = H + J * Delta * ZZ
        for Z in self._Z_terms:
            H = H - h * Z
        
        return H
    
    def ground_state(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute ground state via Lanczos."""
        H = self.hamiltonian(params)
        E, V = eigsh(H, k=1, which='SA')
        return float(E[0]), V[:, 0]
    
    def total_spin_squared(self, params: np.ndarray) -> float:
        """
        Compute ⟨S²⟩ = ⟨(Σ S_i)²⟩.
        
        For the isotropic Heisenberg model, S² is conserved
        and labels the ground state multiplet.
        """
        Sx = self._pauli.total_magnetization('X', self.n_sites) / 2
        Sy = self._pauli.total_magnetization('Y', self.n_sites) / 2
        Sz = self._pauli.total_magnetization('Z', self.n_sites) / 2
        
        S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
        
        _, psi = self.ground_state(params)
        return float(np.real(psi.conj() @ S2 @ psi))
    
    @property
    def critical_point(self) -> dict:
        """Return known critical points for the XXZ chain."""
        return {
            'transitions': [
                {'point': 'Δ = -1', 'type': 'ferro → XY (1st order)'},
                {'point': 'Δ = 1',  'type': 'XY → Néel (BKT transition)'},
            ],
            'phases': {
                'Δ < -1': 'Ferromagnetic',
                '-1 < Δ < 1': 'XY / Luttinger Liquid (gapless)',
                'Δ > 1': 'Néel Antiferromagnet (gapped)',
            }
        }
