#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  benchmark.py — Performance Benchmarking
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_exact_diag():
    """Benchmark exact diagonalization for different system sizes."""
    from qgt.models.ising import IsingModel
    
    print("═══ Exact Diagonalization Benchmark ═══")
    print(f"{'N':>4s} {'dim':>6s} {'Hamiltonian':>12s} {'Eigensolver':>12s} {'QGT':>12s}")
    print("─" * 50)
    
    for N in [4, 6, 8, 10]:
        model = IsingModel(n_sites=N)
        params = np.array([1.0, 0.5])
        
        t0 = time.time()
        H = model.hamiltonian(params)
        t_ham = time.time() - t0
        
        t0 = time.time()
        E, psi = model.ground_state(params)
        t_eig = time.time() - t0
        
        from qgt.core.qgt import QuantumGeometricTensor
        qgt = QuantumGeometricTensor(model.ground_state, 2)
        t0 = time.time()
        chi = qgt.compute(params)
        t_qgt = time.time() - t0
        
        print(f"{N:4d} {2**N:6d} {t_ham*1000:10.1f}ms {t_eig*1000:10.1f}ms {t_qgt*1000:10.1f}ms")


def benchmark_geometry():
    """Benchmark curvature computation."""
    from qgt.models.ising import IsingModel
    from qgt.core.qgt import QuantumGeometricTensor
    from qgt.geometry.metric import MetricTensor
    from qgt.geometry.curvature import CurvatureAnalyzer
    
    print("\n═══ Geometry Computation Benchmark ═══")
    
    model = IsingModel(n_sites=4)
    qgt = QuantumGeometricTensor(model.ground_state, 2)
    metric = MetricTensor(qgt.metric)
    curv = CurvatureAnalyzer(metric)
    
    params = np.array([1.0, 0.5])
    
    t0 = time.time()
    R = curv.scalar_curvature(params)
    t_curv = time.time() - t0
    print(f"Scalar curvature at one point: {t_curv*1000:.1f}ms (R = {R:.4f})")
    
    t0 = time.time()
    scan = curv.scan_curvature([(0.3, 1.7), (0.3, 1.7)], n_points=10)
    t_scan = time.time() - t0
    print(f"10×10 curvature scan: {t_scan:.1f}s")


if __name__ == '__main__':
    benchmark_exact_diag()
    benchmark_geometry()
    print("\n✓ Benchmark complete")
