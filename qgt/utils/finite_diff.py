"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.utils.finite_diff — Numerical Differentiation Utilities
═══════════════════════════════════════════════════════════════════════════════

Provides finite-difference methods for computing derivatives of
quantum states and metric tensors with proper gauge alignment.
"""

import numpy as np
from typing import Callable, Tuple, Optional


def central_difference(f: Callable, x: float, epsilon: float = 1e-5) -> float:
    """
    Central finite difference approximation of df/dx.
    
    f'(x) ≈ [f(x+ε) - f(x-ε)] / (2ε)
    
    Parameters
    ----------
    f : callable
        Scalar or vector-valued function.
    x : float
        Point at which to evaluate the derivative.
    epsilon : float
        Step size.
    
    Returns
    -------
    Derivative value (same type as f output).
    """
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)


def central_difference_2d(f: Callable, x: float, y: float,
                          direction: int, epsilon: float = 1e-5):
    """
    Central finite difference in 2D along a specified direction.
    
    Parameters
    ----------
    f : callable(x, y) → value
        Function of two variables.
    x, y : float
        Point at which to differentiate.
    direction : int
        0 for ∂/∂x, 1 for ∂/∂y.
    epsilon : float
        Step size.
    
    Returns
    -------
    Derivative value.
    """
    if direction == 0:
        return (f(x + epsilon, y) - f(x - epsilon, y)) / (2 * epsilon)
    else:
        return (f(x, y + epsilon) - f(x, y - epsilon)) / (2 * epsilon)


def gradient_2d(f: Callable, x: float, y: float,
                epsilon: float = 1e-5) -> np.ndarray:
    """
    Compute the gradient ∇f = (∂f/∂x, ∂f/∂y) using central differences.
    
    Parameters
    ----------
    f : callable(x, y) → float
    x, y : float
    epsilon : float
    
    Returns
    -------
    ndarray, shape (2,)
    """
    df_dx = central_difference_2d(f, x, y, 0, epsilon)
    df_dy = central_difference_2d(f, x, y, 1, epsilon)
    return np.array([df_dx, df_dy])


def state_derivative(ground_state_fn: Callable,
                     params: np.ndarray,
                     direction: int,
                     epsilon: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂|ψ⟩/∂λ_μ with gauge alignment.
    
    Uses central finite difference with phase alignment to handle
    the U(1) gauge freedom of quantum states.
    
    Parameters
    ----------
    ground_state_fn : callable(params) → (energy, state)
        Function that returns (E, |ψ⟩) given parameters.
    params : ndarray
        Current parameter values.
    direction : int
        Index of the parameter to differentiate w.r.t.
    epsilon : float
        Step size.
    
    Returns
    -------
    dpsi : ndarray
        State derivative vector.
    psi0 : ndarray
        State at the base point.
    """
    _, psi0 = ground_state_fn(params)
    
    params_plus = params.copy()
    params_plus[direction] += epsilon
    _, psi_plus = ground_state_fn(params_plus)
    
    params_minus = params.copy()
    params_minus[direction] -= epsilon
    _, psi_minus = ground_state_fn(params_minus)
    
    # Gauge alignment
    psi_plus = align_phase(psi0, psi_plus)
    psi_minus = align_phase(psi0, psi_minus)
    
    dpsi = (psi_plus - psi_minus) / (2 * epsilon)
    return dpsi, psi0


def align_phase(psi_ref: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Align the global phase of psi to match psi_ref.
    
    Quantum states have gauge freedom |ψ⟩ ~ e^{iφ}|ψ⟩.
    This function multiplies psi by e^{-iφ} where φ = arg(⟨psi_ref|psi⟩).
    
    Parameters
    ----------
    psi_ref : ndarray
        Reference state.
    psi : ndarray
        State to align.
    
    Returns
    -------
    ndarray
        Phase-aligned state.
    """
    overlap = np.vdot(psi_ref, psi)
    if abs(overlap) > 1e-12:
        phase = overlap / abs(overlap)
        return psi * np.conj(phase)
    return psi
