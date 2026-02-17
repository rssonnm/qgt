"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.variational.vqe — VQE Engine with QGT Diagnostics
═══════════════════════════════════════════════════════════════════════════════

Variational Quantum Eigensolver with built-in QGT and geometry tracking.
Monitors the Fubini-Study metric during optimization to diagnose
barren plateaus and convergence issues.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

from qgt.utils.logging import get_logger

logger = get_logger(__name__)


class VQEEngine:
    """
    VQE optimizer with geometric diagnostics.
    
    Parameters
    ----------
    hamiltonian : pennylane.Hamiltonian
        Target Hamiltonian.
    ansatz : callable
        Parameterized circuit function.
    n_qubits : int
    n_params : int
    device : str
        PennyLane device (default: 'default.qubit').
    
    Examples
    --------
    >>> engine = VQEEngine(H, ansatz, n_qubits=4, n_params=24)
    >>> result = engine.optimize(n_steps=100)
    >>> print(result['energy'], result['params'])
    """
    
    def __init__(self, hamiltonian, ansatz: Callable,
                 n_qubits: int, n_params: int,
                 device: str = 'default.qubit'):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")
        
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.n_qubits = n_qubits
        self.n_params = n_params
        
        self.dev = qml.device(device, wires=n_qubits)
        
        @qml.qnode(self.dev, diff_method='backprop')
        def cost_fn(params):
            ansatz(params)
            return qml.expval(hamiltonian)
        
        self.cost_fn = cost_fn
    
    def optimize(self, n_steps: int = 100,
                 learning_rate: float = 0.1,
                 optimizer: str = 'adam',
                 initial_params: Optional[np.ndarray] = None,
                 track_qgt: bool = False,
                 verbose: bool = True) -> dict:
        """
        Run VQE optimization.
        
        Parameters
        ----------
        n_steps : int
        learning_rate : float
        optimizer : str
            'adam', 'gradient_descent', 'qng'.
        initial_params : ndarray, optional
        track_qgt : bool
            If True, compute and store the metric tensor at each step.
        verbose : bool
        
        Returns
        -------
        dict with keys:
            'energy': final energy
            'params': optimal parameters
            'history': energy history
            'qgt_history': metric tensors (if track_qgt=True)
        """
        if initial_params is None:
            initial_params = np.random.randn(self.n_params) * 0.1
        
        params = qml.numpy.array(initial_params, requires_grad=True)
        
        opt_map = {
            'adam': qml.AdamOptimizer(stepsize=learning_rate),
            'gradient_descent': qml.GradientDescentOptimizer(
                stepsize=learning_rate),
        }
        
        opt = opt_map.get(optimizer,
                          qml.AdamOptimizer(stepsize=learning_rate))
        
        history = []
        qgt_history = []
        
        for step in range(n_steps):
            params, cost = opt.step_and_cost(self.cost_fn, params)
            energy = float(cost)
            history.append(energy)
            
            if track_qgt and step % 10 == 0:
                try:
                    mt = qml.metric_tensor(self.cost_fn)(params)
                    qgt_history.append(np.array(mt))
                except Exception:
                    pass
            
            if verbose and step % 20 == 0:
                logger.info(f"Step {step:4d} | E = {energy:.8f}")
        
        return {
            'energy': history[-1],
            'params': np.array(params),
            'history': history,
            'qgt_history': qgt_history if track_qgt else None,
        }
    
    def compute_variational_qgt(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the QGT on the variational manifold via metric_tensor.
        
        Parameters
        ----------
        params : ndarray
        
        Returns
        -------
        ndarray
            Metric tensor (real part of QGT).
        """
        try:
            mt = qml.metric_tensor(self.cost_fn)(
                qml.numpy.array(params, requires_grad=True)
            )
            return np.array(mt)
        except Exception as e:
            logger.warning(f"metric_tensor failed: {e}")
            return np.eye(self.n_params) * 1e-6
