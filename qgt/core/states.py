"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.core.states — Quantum State Representations
═══════════════════════════════════════════════════════════════════════════════

Provides a clean abstraction for quantum states, including:
  - State vector representation
  - Fidelity computation
  - Phase alignment
  - Overlap and inner product utilities

This module underpins all QGT computations by standardizing how
quantum states are created, manipulated, and compared.
"""

import numpy as np
from typing import Optional, Tuple


class QuantumState:
    """
    Represents a normalized quantum state |ψ⟩ in a Hilbert space.
    
    The state is stored as a complex numpy array (state vector).
    Provides methods for computing overlaps, fidelity, and
    gauge-aligned comparisons.
    
    Parameters
    ----------
    amplitudes : ndarray
        Complex amplitudes of the quantum state.
    normalize : bool
        Whether to normalize the state upon creation.
    
    Attributes
    ----------
    dim : int
        Hilbert space dimension.
    vector : ndarray
        The normalized state vector.
    
    Examples
    --------
    >>> psi = QuantumState([1, 0, 0, 1])  # Bell-like state
    >>> psi.norm()
    1.0
    >>> psi.probabilities()
    array([0.5, 0. , 0. , 0.5])
    """
    
    def __init__(self, amplitudes: np.ndarray, normalize: bool = True):
        self.vector = np.asarray(amplitudes, dtype=complex).flatten()
        self.dim = len(self.vector)
        
        if normalize:
            norm = np.linalg.norm(self.vector)
            if norm > 1e-15:
                self.vector /= norm
    
    @classmethod
    def from_label(cls, label: str) -> 'QuantumState':
        """
        Create a computational basis state from a binary string.
        
        Parameters
        ----------
        label : str
            Binary string, e.g. '0010'.
        
        Returns
        -------
        QuantumState
        
        Examples
        --------
        >>> psi = QuantumState.from_label('00')
        >>> psi.vector  # [1, 0, 0, 0]
        """
        n_qubits = len(label)
        dim = 2 ** n_qubits
        idx = int(label, 2)
        vec = np.zeros(dim, dtype=complex)
        vec[idx] = 1.0
        return cls(vec, normalize=False)
    
    @classmethod
    def random(cls, n_qubits: int, seed: Optional[int] = None) -> 'QuantumState':
        """Create a random Haar-distributed quantum state."""
        rng = np.random.default_rng(seed)
        dim = 2 ** n_qubits
        real = rng.standard_normal(dim)
        imag = rng.standard_normal(dim)
        return cls(real + 1j * imag, normalize=True)
    
    def norm(self) -> float:
        """Return the L2 norm ⟨ψ|ψ⟩^{1/2}."""
        return float(np.linalg.norm(self.vector))
    
    def inner_product(self, other: 'QuantumState') -> complex:
        """Compute ⟨self|other⟩."""
        return complex(np.vdot(self.vector, other.vector))
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Compute the fidelity F = |⟨ψ|φ⟩|² between two pure states.
        
        Parameters
        ----------
        other : QuantumState
        
        Returns
        -------
        float
            Fidelity in [0, 1].
        """
        return float(abs(self.inner_product(other)) ** 2)
    
    def probabilities(self) -> np.ndarray:
        """Return the probability distribution |ψ_i|²."""
        return np.abs(self.vector) ** 2
    
    def entropy(self) -> float:
        """Compute the Shannon entropy of the measurement probabilities."""
        probs = self.probabilities()
        # Filter out zeros to avoid log(0)
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))
    
    def align_phase_to(self, reference: 'QuantumState') -> 'QuantumState':
        """
        Return a new state with global phase aligned to the reference.
        
        Fixes the U(1) gauge freedom by multiplying by e^{-iφ}
        where φ = arg(⟨reference|self⟩).
        """
        overlap = np.vdot(reference.vector, self.vector)
        if abs(overlap) > 1e-12:
            phase = overlap / abs(overlap)
            return QuantumState(self.vector * np.conj(phase), normalize=False)
        return QuantumState(self.vector.copy(), normalize=False)
    
    def density_matrix(self) -> np.ndarray:
        """Compute ρ = |ψ⟩⟨ψ| (the pure-state density matrix)."""
        return np.outer(self.vector, np.conj(self.vector))
    
    def reduced_density_matrix(self, keep_qubits: list,
                                n_qubits: int) -> np.ndarray:
        """
        Trace out complementary qubits to get the reduced density matrix.
        
        Parameters
        ----------
        keep_qubits : list of int
            Qubit indices to keep.
        n_qubits : int
            Total number of qubits.
        
        Returns
        -------
        ndarray
            Reduced density matrix.
        """
        # Reshape state into tensor with one index per qubit
        psi_tensor = self.vector.reshape([2] * n_qubits)
        
        # ρ = |ψ⟩⟨ψ| as tensor with 2*n_qubits indices
        # indices: [i0, i1, ..., i_{n-1}, j0, j1, ..., j_{n-1}]
        rho_tensor = np.outer(self.vector, np.conj(self.vector))
        rho_tensor = rho_tensor.reshape([2] * (2 * n_qubits))
        
        trace_qubits = sorted([q for q in range(n_qubits) if q not in keep_qubits],
                               reverse=True)
        
        # Trace over each qubit to remove: contract axis q with axis q + n_remaining
        n_remaining = n_qubits
        for q in trace_qubits:
            # After previous traces, total axes = 2 * n_remaining
            # Qubit q in bra is at axis q, in ket is at axis q + n_remaining
            rho_tensor = np.trace(rho_tensor, axis1=q, axis2=q + n_remaining)
            n_remaining -= 1
        
        n_keep = len(keep_qubits)
        dim_keep = 2 ** n_keep
        return rho_tensor.reshape(dim_keep, dim_keep)
    
    def entanglement_entropy(self, subsystem_qubits: list,
                              n_qubits: int) -> float:
        """
        Compute the von Neumann entanglement entropy S(ρ_A).
        
        S(ρ_A) = -Tr(ρ_A log₂ ρ_A)
        
        Parameters
        ----------
        subsystem_qubits : list
            Qubit indices of subsystem A.
        n_qubits : int
            Total number of qubits.
        
        Returns
        -------
        float
            Von Neumann entropy.
        """
        rho_A = self.reduced_density_matrix(subsystem_qubits, n_qubits)
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    
    def __repr__(self) -> str:
        return f"QuantumState(dim={self.dim}, norm={self.norm():.6f})"
    
    def __eq__(self, other: 'QuantumState') -> bool:
        """Two states are equal if their fidelity is ~1."""
        return self.fidelity(other) > 1 - 1e-10
