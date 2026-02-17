"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.variational.qng — Quantum Natural Gradient Optimizer
═══════════════════════════════════════════════════════════════════════════════

Implements the Quantum Natural Gradient descent:

    θ_{t+1} = θ_t - η · g^{-1}(θ_t) · ∇L(θ_t)

where g is the Fubini-Study metric tensor. This follows the steepest
descent direction in Hilbert space rather than parameter space.

Reference:
    Stokes et al., "Quantum Natural Gradient" (2020)
"""

import numpy as np
from typing import Callable, Optional, List

from qgt.utils.logging import get_logger

logger = get_logger(__name__)


class QuantumNaturalGradient:
    """
    Quantum Natural Gradient optimizer.
    
    Uses the Fubini-Study metric to precondition the gradient,
    achieving faster convergence in variational quantum algorithms.
    
    Parameters
    ----------
    cost_fn : callable(params) → float
        Cost function to minimize.
    metric_fn : callable(params) → ndarray
        Metric tensor function.
    gradient_fn : callable(params) → ndarray
        Gradient function.
    learning_rate : float
    regularization : float
        Tikhonov regularization for metric inversion.
    """
    
    def __init__(self, cost_fn: Callable, metric_fn: Callable,
                 gradient_fn: Callable, learning_rate: float = 0.1,
                 regularization: float = 1e-4):
        self.cost_fn = cost_fn
        self.metric_fn = metric_fn
        self.gradient_fn = gradient_fn
        self.lr = learning_rate
        self.reg = regularization
    
    def step(self, params: np.ndarray) -> tuple:
        """
        Perform one QNG step.
        
        Returns
        -------
        new_params, cost
        """
        g = self.metric_fn(params)
        grad = self.gradient_fn(params)
        cost = self.cost_fn(params)
        
        # Regularized metric inversion
        g_reg = g + self.reg * np.eye(g.shape[0])
        
        try:
            natural_grad = np.linalg.solve(g_reg, grad)
        except np.linalg.LinAlgError:
            logger.warning("Metric inversion failed, falling back to vanilla gradient")
            natural_grad = grad
        
        new_params = params - self.lr * natural_grad
        return new_params, cost
    
    def optimize(self, initial_params: np.ndarray,
                 n_steps: int = 100,
                 verbose: bool = True) -> dict:
        """
        Run QNG optimization.
        
        Returns
        -------
        dict with keys:
            'params': optimal parameters
            'cost_history': cost at each step
            'gradient_norms': Euclidean gradient norms
            'natural_gradient_norms': QNG gradient norms
            'metric_condition_numbers': condition number of g at each step
        """
        params = np.copy(initial_params)
        history = {
            'params_history': [params.copy()],
            'cost_history': [],
            'gradient_norms': [],
            'natural_gradient_norms': [],
            'metric_condition_numbers': [],
        }
        
        for step in range(n_steps):
            g = self.metric_fn(params)
            grad = self.gradient_fn(params)
            cost = self.cost_fn(params)
            
            g_reg = g + self.reg * np.eye(g.shape[0])
            cond = np.linalg.cond(g_reg)
            
            try:
                nat_grad = np.linalg.solve(g_reg, grad)
            except np.linalg.LinAlgError:
                nat_grad = grad
            
            params = params - self.lr * nat_grad
            
            history['cost_history'].append(float(cost))
            history['gradient_norms'].append(float(np.linalg.norm(grad)))
            history['natural_gradient_norms'].append(
                float(np.linalg.norm(nat_grad)))
            history['metric_condition_numbers'].append(float(cond))
            history['params_history'].append(params.copy())
            
            if verbose and step % 20 == 0:
                logger.info(
                    f"Step {step:4d} | cost={cost:.8f} | "
                    f"|∇|={np.linalg.norm(grad):.4e} | "
                    f"|g⁻¹∇|={np.linalg.norm(nat_grad):.4e} | "
                    f"κ(g)={cond:.1f}"
                )
        
        history['params'] = params
        return history
    
    @staticmethod
    def compare_with_vanilla(cost_fn, metric_fn, gradient_fn,
                              initial_params, n_steps=100,
                              qng_lr=0.05, gd_lr=0.1) -> dict:
        """
        Compare QNG vs vanilla gradient descent convergence.
        
        Returns
        -------
        dict with 'qng_history' and 'gd_history'
        """
        # QNG
        qng = QuantumNaturalGradient(
            cost_fn, metric_fn, gradient_fn,
            learning_rate=qng_lr
        )
        qng_result = qng.optimize(initial_params.copy(), n_steps,
                                    verbose=False)
        
        # Vanilla GD
        params = initial_params.copy()
        gd_history = []
        for _ in range(n_steps):
            cost = cost_fn(params)
            grad = gradient_fn(params)
            gd_history.append(float(cost))
            params = params - gd_lr * grad
        
        return {
            'qng_history': qng_result['cost_history'],
            'gd_history': gd_history,
        }
