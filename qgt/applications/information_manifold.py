"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.applications.information_manifold — Full Manifold Analysis
═══════════════════════════════════════════════════════════════════════════════

End-to-end pipeline for computing and visualizing the information manifold
of a parameterized quantum model. Orchestrates QGT computation, geometry
analysis, and visualization into a single coherent workflow.
"""

import numpy as np
import time
from typing import Optional, List

from qgt.core.qgt import QuantumGeometricTensor
from qgt.geometry.metric import MetricTensor
from qgt.geometry.curvature import CurvatureAnalyzer
from qgt.geometry.geodesics import GeodesicSolver
from qgt.geometry.topology import TopologicalInvariants
from qgt.models.base import QuantumModel
from qgt.utils.logging import get_logger

logger = get_logger(__name__)


class InformationManifoldAnalyzer:
    """
    Comprehensive analysis of the quantum information manifold.
    
    Brings together QGT, curvature, geodesics, and topology into
    a single pipeline. Designed for research-grade analysis of
    quantum phase transitions and information geometry.
    
    Parameters
    ----------
    model : QuantumModel
        The physical model to analyze.
    
    Examples
    --------
    >>> model = IsingModel(n_sites=4)
    >>> analyzer = InformationManifoldAnalyzer(model)
    >>> results = analyzer.full_analysis(
    ...     param_ranges=[(0.1, 2.0), (0.1, 2.0)],
    ...     n_points=25
    ... )
    """
    
    def __init__(self, model: QuantumModel):
        self.model = model
        self.qgt = QuantumGeometricTensor(
            model.ground_state, model.n_params
        )
        self._metric = MetricTensor(self.qgt.metric)
        self._curvature = CurvatureAnalyzer(self._metric)
        self._geodesic_solver = GeodesicSolver(self._metric)
        self._topology = TopologicalInvariants(self.qgt.berry_curvature)
    
    def full_analysis(self, param_ranges: List[tuple],
                       n_points: int = 25,
                       compute_curvature: bool = True,
                       compute_geodesics: bool = True,
                       n_curvature_pts: int = 15,
                       n_geodesic_grid: int = 12) -> dict:
        """
        Run a complete geometric analysis of the information manifold.
        
        Parameters
        ----------
        param_ranges : list of (min, max)
        n_points : int
            Grid density for QGT scan.
        compute_curvature : bool
        compute_geodesics : bool
        n_curvature_pts : int
            Grid density for curvature (computationally expensive).
        n_geodesic_grid : int
            Grid density for Christoffel pre-computation.
        
        Returns
        -------
        dict
            Comprehensive results including all geometric quantities.
        """
        t0 = time.time()
        results = {}
        
        # 1. QGT scan
        logger.info("Scanning parameter space for QGT...")
        qgt_scan = self.qgt.scan_parameter_space(
            param_ranges, n_points,
            quantities=['metric_det', 'metric_trace', 'berry_curv',
                        'fidelity_sus']
        )
        results['qgt_scan'] = qgt_scan
        logger.info(f"  QGT scan: {time.time() - t0:.1f}s")
        
        # 2. Energy gap
        logger.info("Scanning energy gap...")
        results['gap_scan'] = self.model.scan_energy_gap(
            param_ranges, n_points
        )
        
        # 3. Scalar curvature
        if compute_curvature:
            logger.info("Computing scalar curvature...")
            t1 = time.time()
            results['curvature'] = self._curvature.scan_curvature(
                param_ranges, n_curvature_pts
            )
            logger.info(f"  Curvature: {time.time() - t1:.1f}s")
        
        # 4. Geodesics
        if compute_geodesics:
            logger.info("Computing geodesics...")
            t2 = time.time()
            self._geodesic_solver.precompute_grid(
                param_ranges, n_geodesic_grid
            )
            
            # Generate several geodesics from different starting points
            geodesic_paths = []
            center = [(r[0] + r[1]) / 2 for r in param_ranges]
            
            # Geodesics radiating from a point in the ferromagnetic phase
            start_pt = np.array([param_ranges[0][0] + 0.3,
                                  param_ranges[1][0] + 0.3])
            for angle in np.linspace(0, np.pi, 5):
                v0 = 0.3 * np.array([np.cos(angle), np.sin(angle)])
                path = self._geodesic_solver.solve(start_pt, v0)
                geodesic_paths.append(path)
            
            results['geodesics'] = geodesic_paths
            logger.info(f"  Geodesics: {time.time() - t2:.1f}s")
        
        # 5. Chern number
        try:
            results['chern_number'] = self._topology.chern_number(
                param_ranges, n_points=30
            )
            logger.info(f"  Chern number: {results['chern_number']:.4f}")
        except Exception:
            results['chern_number'] = None
        
        results['total_time'] = time.time() - t0
        logger.info(f"Total analysis time: {results['total_time']:.1f}s")
        
        return results
    
    def cross_section(self, fixed_param_idx: int,
                       fixed_value: float,
                       scan_range: tuple,
                       n_points: int = 50) -> dict:
        """
        Compute a 1D cross-section of geometric quantities.
        
        Parameters
        ----------
        fixed_param_idx : int
            Index of the fixed parameter.
        fixed_value : float
        scan_range : (min, max)
        n_points : int
        
        Returns
        -------
        dict with 1D arrays of geometric quantities.
        """
        scan_vals = np.linspace(scan_range[0], scan_range[1], n_points)
        scan_idx = 1 - fixed_param_idx
        
        results = {
            'scan_values': scan_vals,
            'fidelity_sus': np.zeros(n_points),
            'metric_det': np.zeros(n_points),
            'berry_curv': np.zeros(n_points),
            'energy_gap': np.zeros(n_points),
        }
        
        for i, val in enumerate(scan_vals):
            params = np.zeros(self.model.n_params)
            params[fixed_param_idx] = fixed_value
            params[scan_idx] = val
            
            try:
                chi = self.qgt.compute(params)
                g = np.real(chi)
                results['fidelity_sus'][i] = np.trace(g) / self.model.n_sites
                results['metric_det'][i] = np.linalg.det(g)
                results['berry_curv'][i] = 2 * np.imag(chi[0, 1])
                results['energy_gap'][i] = self.model.energy_gap(params)
            except Exception:
                for key in ['fidelity_sus', 'metric_det', 'berry_curv',
                             'energy_gap']:
                    results[key][i] = np.nan
        
        return results
