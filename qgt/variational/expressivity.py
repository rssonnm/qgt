"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.variational.expressivity — Ansatz Expressivity Analysis
═══════════════════════════════════════════════════════════════════════════════

Quantifies how well a variational ansatz captures the true geometry
of the quantum state manifold. Uses QGT-based metrics:
  - Metric trace ratio (variational vs exact)  
  - Fidelity susceptibility mismatch
  - Effective dimension of the variational manifold
"""

import numpy as np
from typing import Callable, Tuple

from qgt.utils.logging import get_logger

logger = get_logger(__name__)


class ExpressivityAnalyzer:
    """
    Analyze ansatz expressivity through geometric quantities.
    
    Compares the geometry of the exact ground state manifold
    to the variational manifold to assess how faithfully the
    ansatz captures the information-geometric structure.
    
    Parameters
    ----------
    exact_qgt_fn : callable(params) → ndarray (n×n)
        QGT on the exact manifold.
    variational_qgt_fn : callable(params) → ndarray (m×m)
        QGT on the variational manifold.
    n_sites : int
        Number of sites for normalization.
    """
    
    def __init__(self, exact_qgt_fn: Callable,
                 variational_qgt_fn: Callable,
                 n_sites: int = 1):
        self.exact_fn = exact_qgt_fn
        self.var_fn = variational_qgt_fn
        self.n_sites = n_sites
    
    def fidelity_susceptibility_ratio(self, params: np.ndarray) -> float:
        """
        Compute χ_F^{var} / χ_F^{exact}.
        
        Ratio < 1 means the ansatz underestimates the
        sensitivity of the ground state to parameter changes.
        """
        g_exact = np.real(self.exact_fn(params))
        g_var = np.real(self.var_fn(params))
        
        chi_exact = np.trace(g_exact) / self.n_sites
        chi_var = np.trace(g_var) / self.n_sites
        
        if chi_exact < 1e-15:
            return 0.0
        return float(chi_var / chi_exact)
    
    def effective_dimension(self, params: np.ndarray,
                             threshold: float = 1e-6) -> int:
        """
        Compute the effective dimension of the variational manifold.
        
        This is the number of eigenvalues of the variational metric
        that exceed the threshold. A low effective dimension indicates
        redundant parameters (barren plateau risk).
        
        Parameters
        ----------
        params : ndarray
        threshold : float
        
        Returns
        -------
        int
        """
        g = np.real(self.var_fn(params))
        eigenvalues = np.linalg.eigvalsh(g)
        return int(np.sum(eigenvalues > threshold))
    
    def metric_spectrum_comparison(self, params: np.ndarray) -> dict:
        """
        Compare eigenvalue spectra of exact and variational metrics.
        
        Returns
        -------
        dict with 'exact_spectrum', 'var_spectrum', 'spectral_gap_ratio'
        """
        g_e = np.real(self.exact_fn(params))
        g_v = np.real(self.var_fn(params))
        
        spec_e = np.sort(np.linalg.eigvalsh(g_e))[::-1]
        spec_v = np.sort(np.linalg.eigvalsh(g_v))[::-1]
        
        # Spectral gap ratio for the leading eigenvalue
        ratio = spec_v[0] / spec_e[0] if spec_e[0] > 1e-15 else 0
        
        return {
            'exact_spectrum': spec_e,
            'var_spectrum': spec_v,
            'spectral_gap_ratio': float(ratio),
        }
    
    def scan_expressivity(self, param_values: np.ndarray,
                           fixed_params: dict = None) -> dict:
        """
        Scan expressivity metrics across a range of one parameter.
        
        Parameters
        ----------
        param_values : ndarray
            Values of the scanning parameter.
        fixed_params : dict
            Other fixed parameters {index: value}.
        
        Returns
        -------
        dict with arrays of metrics at each point.
        """
        results = {
            'params': param_values,
            'chi_ratio': np.zeros(len(param_values)),
            'eff_dim': np.zeros(len(param_values), dtype=int),
        }
        
        for i, val in enumerate(param_values):
            if fixed_params:
                params = np.zeros(2)
                for k, v in fixed_params.items():
                    params[k] = v
                # Assume scanning the first unfixed parameter
                unfixed = [k for k in range(2) if k not in fixed_params]
                if unfixed:
                    params[unfixed[0]] = val
            else:
                params = np.array([val])
            
            results['chi_ratio'][i] = self.fidelity_susceptibility_ratio(params)
            results['eff_dim'][i] = self.effective_dimension(params)
        
        return results
