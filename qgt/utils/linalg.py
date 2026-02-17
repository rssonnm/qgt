"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.utils.linalg — Linear Algebra Helpers
═══════════════════════════════════════════════════════════════════════════════

Utility functions for sparse linear algebra, tensor products, 
and safe matrix operations used throughout the QGT package.
"""

import numpy as np
from scipy.sparse import kron as sp_kron, eye as sp_eye, csr_matrix
from scipy.sparse.linalg import eigsh
from typing import List, Tuple, Optional


def sparse_kron_chain(operators: List[csr_matrix]) -> csr_matrix:
    """
    Compute the tensor product of a chain of sparse operators.
    
    Parameters
    ----------
    operators : list of csr_matrix
        Operators to tensor-product together.
    
    Returns
    -------
    csr_matrix
        The full tensor product operator.
    
    Example
    -------
    >>> I = sparse_identity(2)
    >>> Z = csr_matrix(np.diag([1, -1]))
    >>> IZI = sparse_kron_chain([I, Z, I])
    """
    result = operators[0]
    for op in operators[1:]:
        result = sp_kron(result, op, format='csr')
    return result


def sparse_identity(dim: int) -> csr_matrix:
    """Sparse identity matrix."""
    return sp_eye(dim, format='csr')


def embed_operator(op: csr_matrix, site: int, n_sites: int,
                   local_dim: int = 2) -> csr_matrix:
    """
    Embed a single-site operator into a multi-site Hilbert space.
    
    Constructs: I^{⊗site} ⊗ op ⊗ I^{⊗(n_sites - site - 1)}
    
    Parameters
    ----------
    op : csr_matrix
        Local operator (local_dim × local_dim).
    site : int
        Site index (0-based).
    n_sites : int
        Total number of sites.
    local_dim : int
        Local Hilbert space dimension (default: 2 for qubits).
    
    Returns
    -------
    csr_matrix
        Operator embedded in the full Hilbert space.
    """
    I = sparse_identity(local_dim)
    ops = [I] * n_sites
    ops[site] = op
    return sparse_kron_chain(ops)


def ground_state(H: csr_matrix, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the k lowest eigenstates of a sparse Hamiltonian.
    
    Uses Lanczos algorithm (scipy.sparse.linalg.eigsh).
    
    Parameters
    ----------
    H : csr_matrix
        Hamiltonian matrix.
    k : int
        Number of eigenvalues to compute.
    
    Returns
    -------
    energies : ndarray, shape (k,)
    states : ndarray, shape (dim, k)
    """
    E, V = eigsh(H, k=k, which='SA')
    # Sort by energy
    idx = np.argsort(E)
    return E[idx], V[:, idx]


def matrix_function_safe(A: np.ndarray, func=np.linalg.inv,
                         reg: float = 1e-10) -> np.ndarray:
    """
    Apply a matrix function with regularization for numerical stability.
    
    Parameters
    ----------
    A : ndarray
        Input matrix.
    func : callable
        Matrix function (default: inverse).
    reg : float
        Regularization parameter added to diagonal.
    
    Returns
    -------
    ndarray
        Result of func(A + reg * I).
    """
    A_reg = A + reg * np.eye(A.shape[0])
    return func(A_reg)
