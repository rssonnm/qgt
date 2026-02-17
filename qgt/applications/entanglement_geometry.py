"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.applications.entanglement_geometry — Entanglement ↔ Geometry Link
═══════════════════════════════════════════════════════════════════════════════

Explores the connection between quantum entanglement and the geometry
of the information manifold:
  - Entanglement entropy as a function of model parameters
  - Correlation between entanglement and QGT quantities
  - Entanglement area law and its violations at criticality
"""

import numpy as np
from typing import Callable, List, Optional

from qgt.core.states import QuantumState
from qgt.models.base import QuantumModel
from qgt.utils.logging import get_logger

logger = get_logger(__name__)


class EntanglementGeometryAnalyzer:
    """
    Studies the relationship between entanglement and information geometry.
    
    At quantum phase transitions, both entanglement entropy and
    the Fubini-Study metric diverge. This analyzer quantifies
    this connection.
    
    Parameters
    ----------
    model : QuantumModel
    """
    
    def __init__(self, model: QuantumModel):
        self.model = model
    
    def entanglement_entropy_scan(self, param_ranges: list,
                                   n_points: int = 20,
                                   bipartition: Optional[list] = None) -> dict:
        """
        Scan the half-chain entanglement entropy S(ρ_A).
        
        Parameters
        ----------
        param_ranges : list of (min, max) tuples
        n_points : int
        bipartition : list of int, optional
            Qubit indices for subsystem A.
            Default: first half of the chain.
        
        Returns
        -------
        dict with 'param_0', 'param_1', 'entropy' arrays
        """
        if bipartition is None:
            bipartition = list(range(self.model.n_sites // 2))
        
        p0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points)
        p1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points)
        entropy = np.zeros((n_points, n_points))
        
        for i, x in enumerate(p0):
            for j, y in enumerate(p1):
                _, state = self.model.ground_state(np.array([x, y]))
                qs = QuantumState(state, normalize=False)
                try:
                    entropy[i, j] = qs.entanglement_entropy(
                        bipartition, self.model.n_sites
                    )
                except Exception:
                    entropy[i, j] = np.nan
        
        return {'param_0': p0, 'param_1': p1, 'entropy': entropy}
    
    def entropy_geometry_correlation(self, scan_param: int,
                                      scan_range: tuple,
                                      fixed_params: dict,
                                      n_points: int = 30) -> dict:
        """
        Compute the correlation between entanglement entropy
        and fidelity susceptibility along a parameter cut.
        
        Returns
        -------
        dict with 1D arrays and Pearson correlation coefficient.
        """
        from qgt.core.qgt import QuantumGeometricTensor
        
        qgt = QuantumGeometricTensor(
            self.model.ground_state, self.model.n_params
        )
        bipartition = list(range(self.model.n_sites // 2))
        
        scan_vals = np.linspace(scan_range[0], scan_range[1], n_points)
        entropies = np.zeros(n_points)
        chi_F = np.zeros(n_points)
        
        for i, val in enumerate(scan_vals):
            params = np.zeros(self.model.n_params)
            for k, v in fixed_params.items():
                params[k] = v
            params[scan_param] = val
            
            _, state = self.model.ground_state(params)
            qs = QuantumState(state, normalize=False)
            
            try:
                entropies[i] = qs.entanglement_entropy(
                    bipartition, self.model.n_sites
                )
                g = qgt.metric(params)
                chi_F[i] = np.trace(g) / self.model.n_sites
            except Exception:
                entropies[i] = np.nan
                chi_F[i] = np.nan
        
        # Pearson correlation
        valid = ~(np.isnan(entropies) | np.isnan(chi_F))
        if valid.sum() > 2:
            corr = np.corrcoef(entropies[valid], chi_F[valid])[0, 1]
        else:
            corr = np.nan
        
        return {
            'scan_values': scan_vals,
            'entanglement_entropy': entropies,
            'fidelity_susceptibility': chi_F,
            'correlation': float(corr),
        }
