"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.applications.phase_transitions — QPT Detection via Geometry
═══════════════════════════════════════════════════════════════════════════════

Automated detection of quantum phase transitions using geometric
singularities in the QGT. Provides tools for:
  - Peak detection in fidelity susceptibility
  - Scaling analysis near critical points
  - Finite-size scaling of geometric quantities
"""

import numpy as np
from typing import List, Tuple, Optional

from qgt.core.qgt import QuantumGeometricTensor
from qgt.models.base import QuantumModel
from qgt.utils.logging import get_logger

logger = get_logger(__name__)


class PhaseTransitionDetector:
    """
    Detect quantum phase transitions via geometric singularities.
    
    Uses the QGT to identify critical points where the fidelity
    susceptibility (Tr(g)/N), metric determinant, or scalar curvature
    diverge.
    
    Parameters
    ----------
    model : QuantumModel
        The quantum model to analyze.
    
    Examples
    --------
    >>> model = IsingModel(n_sites=6)
    >>> detector = PhaseTransitionDetector(model)
    >>> result = detector.find_critical_points(
    ...     scan_param=1,
    ...     scan_range=(0.1, 2.0),
    ...     fixed_params={0: 1.0}
    ... )
    """
    
    def __init__(self, model: QuantumModel):
        self.model = model
        self.qgt = QuantumGeometricTensor(
            model.ground_state, model.n_params
        )
    
    def find_critical_points(self, scan_param: int,
                              scan_range: Tuple[float, float],
                              fixed_params: dict,
                              n_points: int = 50,
                              metric: str = 'fidelity_sus') -> dict:
        """
        Scan one parameter while fixing others to find where geometric
        quantities peak (indicating phase transitions).
        
        Parameters
        ----------
        scan_param : int
            Index of the parameter to scan.
        scan_range : (min, max)
        fixed_params : dict
            {param_index: value} for fixed parameters.
        n_points : int
        metric : str
            'fidelity_sus', 'metric_det', 'energy_gap'
        
        Returns
        -------
        dict with:
            'scan_values': parameter values
            'metric_values': computed metric at each point
            'critical_point': estimated critical value
            'peak_value': maximum metric value
        """
        scan_vals = np.linspace(scan_range[0], scan_range[1], n_points)
        metric_vals = np.zeros(n_points)
        
        for i, val in enumerate(scan_vals):
            params = np.zeros(self.model.n_params)
            for k, v in fixed_params.items():
                params[k] = v
            params[scan_param] = val
            
            try:
                if metric == 'fidelity_sus':
                    metric_vals[i] = self.qgt.fidelity_susceptibility(
                        params, self.model.n_sites)
                elif metric == 'metric_det':
                    metric_vals[i] = self.qgt.metric_determinant(params)
                elif metric == 'energy_gap':
                    metric_vals[i] = self.model.energy_gap(params)
            except Exception:
                metric_vals[i] = np.nan
        
        # Find critical point (peak or minimum gap)
        valid = ~np.isnan(metric_vals)
        if metric == 'energy_gap':
            critical_idx = np.argmin(metric_vals[valid])
        else:
            critical_idx = np.argmax(metric_vals[valid])
        
        critical_val = scan_vals[valid][critical_idx]
        
        return {
            'scan_values': scan_vals,
            'metric_values': metric_vals,
            'critical_point': float(critical_val),
            'peak_value': float(metric_vals[valid][critical_idx]),
            'metric_name': metric,
        }
    
    def finite_size_scaling(self, system_sizes: List[int],
                             scan_param: int,
                             scan_range: Tuple[float, float],
                             fixed_params: dict,
                             n_points: int = 30) -> dict:
        """
        Finite-size scaling of the fidelity susceptibility.
        
        The peak position shifts as N^{-1/ν} and the peak height
        grows as N^{2/ν - d} for a d-dimensional QPT with exponent ν.
        
        Parameters
        ----------
        system_sizes : list of int
        scan_param, scan_range, fixed_params : as in find_critical_points
        
        Returns
        -------
        dict with scaling data for each system size.
        """
        results = {'sizes': system_sizes}
        peaks = []
        peak_positions = []
        
        for N in system_sizes:
            logger.info(f"Running N={N}...")
            model_N = self.model.__class__(n_sites=N)
            qgt_N = QuantumGeometricTensor(
                model_N.ground_state, model_N.n_params
            )
            
            scan_vals = np.linspace(scan_range[0], scan_range[1], n_points)
            chi_F = np.zeros(n_points)
            
            for i, val in enumerate(scan_vals):
                params = np.zeros(model_N.n_params)
                for k, v in fixed_params.items():
                    params[k] = v
                params[scan_param] = val
                
                try:
                    chi_F[i] = qgt_N.fidelity_susceptibility(params, N)
                except Exception:
                    chi_F[i] = np.nan
            
            valid = ~np.isnan(chi_F)
            peak_idx = np.argmax(chi_F[valid])
            peaks.append(float(chi_F[valid][peak_idx]))
            peak_positions.append(float(scan_vals[valid][peak_idx]))
        
        results['peak_heights'] = peaks
        results['peak_positions'] = peak_positions
        return results
    
    def scaling_exponent(self, sizes: np.ndarray,
                          peak_heights: np.ndarray) -> float:
        """
        Estimate the critical exponent ν from finite-size scaling.
        
        χ_F^{peak} ~ N^{2/ν} → log(χ_F) ~ (2/ν) log(N)
        
        Returns
        -------
        float
            Estimated ν.
        """
        log_N = np.log(np.array(sizes, dtype=float))
        log_chi = np.log(np.array(peak_heights, dtype=float))
        
        # Linear fit
        slope, _ = np.polyfit(log_N, log_chi, 1)
        
        # slope ≈ 2/ν → ν ≈ 2/slope
        if abs(slope) > 1e-10:
            return 2.0 / slope
        return float('inf')
