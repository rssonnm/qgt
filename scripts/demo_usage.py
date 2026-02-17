#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  QGT Package — Demo Sử Dụng Toàn Diện
═══════════════════════════════════════════════════════════════════════════════

Hướng dẫn cách sử dụng tất cả các module của package qgt/
từ cơ bản → nâng cao.
"""

import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

output_dir = 'output/demo'
os.makedirs(output_dir, exist_ok=True)


def separator(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 1: Core — Quantum States & Operators
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 1: Core — Quantum States & Operators")

# --- 1.1 Quantum States ---
from qgt.core.states import QuantumState

# Tạo state từ vector
psi = QuantumState([1, 0, 0, 1])  # Bell state |00⟩ + |11⟩ (tự normalize)
print(f"Bell state: {psi}")
print(f"  Norm       = {psi.norm():.4f}")
print(f"  Entropy    = {psi.entropy():.4f} bits")
print(f"  Probs      = {psi.probabilities()}")

# Tạo state từ label
zero = QuantumState.from_label('00')
print(f"\n|00⟩ state: {zero.vector}")

# Fidelity (mức giống nhau)
phi = QuantumState.random(n_qubits=2, seed=42)
print(f"\nFidelity ⟨Bell|random⟩ = {psi.fidelity(phi):.4f}")
print(f"Fidelity ⟨Bell|Bell⟩  = {psi.fidelity(psi):.4f}")

# Entanglement Entropy (chia nửa hệ)
S = psi.entanglement_entropy(subsystem_qubits=[0], n_qubits=2)
print(f"\nEntanglement entropy (half-chain): S = {S:.4f}")
print("  → S ≈ 1.0 = maximally entangled! ✓")

# --- 1.2 Pauli Operators ---
from qgt.core.operators import PauliOperators

pauli = PauliOperators()
print(f"\nPauli X =\n{pauli.X}")
print(f"\n[X, Y] = 2iZ? → {np.allclose(pauli.X @ pauli.Y - pauli.Y @ pauli.X, 2j * pauli.Z)}")

# N-body operator: Z₀ trên hệ 4 qubit
Z0 = pauli.n_body_operator({0: pauli.Z}, n_sites=4)
print(f"\nZ₀ on 4 qubits: shape = {Z0.shape}")

# --- 1.3 Quantum Geometric Tensor ---
from qgt.core.qgt import QuantumGeometricTensor

# Ví dụ đơn giản: single-qubit rotation
# |ψ(θ)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
# QGT lý thuyết: g = 1/4
def single_qubit_state(params):
    theta = params[0]
    psi = np.array([np.cos(theta/2), np.sin(theta/2)])
    return 0.0, psi

qgt = QuantumGeometricTensor(single_qubit_state, n_params=1)
g = qgt.metric(np.array([1.0]))
print(f"\n--- QGT for single-qubit rotation ---")
print(f"Metric g(θ=1) = {g[0,0]:.6f}")
print(f"Lý thuyết:    g = 0.250000")
print(f"Sai số:       {abs(g[0,0] - 0.25):.2e}")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 2: Models — Physical Systems
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 2: Models — Ising, Heisenberg, SU(2)")

# --- 2.1 Ising Model ---
from qgt.models.ising import IsingModel

model = IsingModel(n_sites=6, periodic=True)
print(f"Model: {model}")
print(f"Critical point: {model.critical_point['condition']}")

# Tính ground state
params_ferro = np.array([1.0, 0.3])  # Deep ferromagnetic (h << J)
params_crit  = np.array([1.0, 1.0])  # Critical point (h = J)
params_para  = np.array([1.0, 2.0])  # Deep paramagnetic (h >> J)

E_f, _ = model.ground_state(params_ferro)
E_c, _ = model.ground_state(params_crit)
E_p, _ = model.ground_state(params_para)

print(f"\n  Ferro  (J=1, h=0.3): E = {E_f:.4f}")
print(f"  Critical (J=1, h=1): E = {E_c:.4f}")
print(f"  Para  (J=1, h=2.0): E = {E_p:.4f}")

# Energy gap — đóng lại ở critical!
gap_f = model.energy_gap(params_ferro)
gap_c = model.energy_gap(params_crit)
gap_p = model.energy_gap(params_para)
print(f"\n  Energy gap:")
print(f"    Ferro:    Δ = {gap_f:.4f}")
print(f"    Critical: Δ = {gap_c:.4f}  ← nhỏ nhất!")
print(f"    Para:     Δ = {gap_p:.4f}")

# --- 2.2 Heisenberg Model ---
from qgt.models.heisenberg import HeisenbergModel

heisenberg = HeisenbergModel(n_sites=4)
print(f"\n{heisenberg}")
E_iso, _ = heisenberg.ground_state(np.array([1.0, 1.0, 0.0]))
print(f"Isotropic (Δ=1): E = {E_iso:.4f}")

# --- 2.3 SU(2) J1-J2 Model ---
from qgt.models.su2_model import SU2Model

su2 = SU2Model(n_sites=4)
print(f"\n{su2}")
E_mg, _ = su2.ground_state(np.array([1.0, 0.5]))
print(f"Majumdar-Ghosh (J2/J1=0.5): E = {E_mg:.4f}")
D = su2.dimer_order_parameter(np.array([1.0, 0.5]))
print(f"Dimer order parameter: D = {D:.4f}")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 3: QGT trên Model Vật Lý
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 3: QGT trên Model Vật Lý")

model4 = IsingModel(n_sites=4)
qgt_ising = QuantumGeometricTensor(model4.ground_state, n_params=2)

# Tính QGT tại 1 điểm
chi = qgt_ising.compute(np.array([1.0, 0.5]))
g = qgt_ising.metric(np.array([1.0, 0.5]))
omega = qgt_ising.berry_curvature(np.array([1.0, 0.5]))

print("QGT tại (J=1, h=0.5):")
print(f"  Full QGT χ =\n{chi}")
print(f"\n  Fubini-Study metric g (real part):")
print(f"    g_JJ = {g[0,0]:.6f}")
print(f"    g_Jh = {g[0,1]:.6f}")
print(f"    g_hh = {g[1,1]:.6f}")
print(f"\n  Berry curvature Ω₁₂ = {omega[0,1]:.6f}")

# Fidelity susceptibility — đỉnh tại QPT!
print("\n  Fidelity susceptibility (per site) scan:")
for h in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    chi_F = qgt_ising.fidelity_susceptibility(np.array([1.0, h]), n_sites=4)
    bar = '█' * int(chi_F * 20)
    print(f"    h={h:.1f}: χF = {chi_F:.4f}  {bar}")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 4: Geometry — Curvature & Geodesics
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 4: Geometry — Metric, Curvature, Connections")

from qgt.geometry.metric import MetricTensor
from qgt.geometry.connection import ChristoffelSymbols
from qgt.geometry.curvature import CurvatureAnalyzer

metric = MetricTensor(qgt_ising.metric)
chris = ChristoffelSymbols(metric)
curv = CurvatureAnalyzer(metric)

# Metric properties
point = np.array([1.0, 0.5])
print(f"Tại (J=1, h=0.5):")
print(f"  det(g)           = {metric.determinant(point):.6f}")
print(f"  Condition number = {metric.condition_number(point):.2f}")
print(f"  Volume element   = {metric.volume_element(point):.6f}")

# Christoffel symbols
Gamma = chris.compute(point)
print(f"\n  Christoffel symbols Γ^σ_μν:")
for s in range(2):
    for m in range(2):
        for n in range(m, 2):
            if abs(Gamma[s, m, n]) > 1e-6:
                print(f"    Γ^{s}_{m}{n} = {Gamma[s,m,n]:.4f}")

# Scalar curvature
R = curv.scalar_curvature(point)
print(f"\n  Scalar curvature R = {R:.2f}")

# Curvature near critical point
print(f"\n  Curvature scan along h (J=1 fixed):")
for h in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
    try:
        R_h = curv.scalar_curvature(np.array([1.0, h]))
        print(f"    h={h:.1f}: R = {R_h:.1f}")
    except Exception:
        print(f"    h={h:.1f}: R = diverges")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 5: Topology — Chern Number & Berry Phase
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 5: Topology — Berry Phase & Chern Number")

from qgt.geometry.topology import TopologicalInvariants

topo = TopologicalInvariants(qgt_ising.berry_curvature)

# Berry phase quanh vòng
t_vals = np.linspace(0, 1, 101)
path = np.array([[1.0 + 0.3*np.cos(2*np.pi*t),
                   1.0 + 0.3*np.sin(2*np.pi*t)] for t in t_vals])
phase = topo.berry_phase_loop(model4.ground_state, path)
print(f"Berry phase (quanh vòng r=0.3, tâm critical): γ = {phase:.4f}")

# Chern number
C = topo.chern_number([(0.3, 1.7), (0.3, 1.7)], n_points=30)
print(f"Chern number: C₁ = {C:.4f}")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 6: Applications — Phase Transition Detection
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 6: Ứng Dụng — Phát Hiện QPT Tự Động")

from qgt.applications.phase_transitions import PhaseTransitionDetector

detector = PhaseTransitionDetector(IsingModel(n_sites=6))
result = detector.find_critical_points(
    scan_param=1,
    scan_range=(0.1, 2.0),
    fixed_params={0: 1.0},
    n_points=50,
    metric='fidelity_sus',
)

print(f"Tự động phát hiện phase transition:")
print(f"  Critical point: h_c = {result['critical_point']:.3f}")
print(f"  Peak χ_F = {result['peak_value']:.4f}")
print(f"  (Lý thuyết: h_c = 1.000)")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 7: Visualization — Tạo Plots
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 7: Visualization — Publication-Quality Plots")

import matplotlib
matplotlib.use('Agg')

# Scan parameter space
print("Scanning 20×20 parameter space...")
t0 = time.time()
qgt_scan = qgt_ising.scan_parameter_space(
    [(0.2, 2.0), (0.2, 2.0)], n_points=20,
    quantities=['metric_det', 'metric_trace', 'berry_curv', 'fidelity_sus']
)
print(f"  Done in {time.time()-t0:.1f}s")

# Plot
from qgt.visualization.manifold_plots import ManifoldPlotter

plotter = ManifoldPlotter(output_dir)
plotter.plot_overview(qgt_scan, save_name='demo_overview')
print(f"  Saved: {output_dir}/demo_overview.png")

# Cross-section
from qgt.applications.information_manifold import InformationManifoldAnalyzer

analyzer = InformationManifoldAnalyzer(model4)
cross = analyzer.cross_section(
    fixed_param_idx=0, fixed_value=1.0,
    scan_range=(0.2, 2.0), n_points=40
)
plotter.plot_cross_section(cross, save_name='demo_cross_section')
print(f"  Saved: {output_dir}/demo_cross_section.png")


# ═══════════════════════════════════════════════════════════════
#  LEVEL 8: Variational (PennyLane)
# ═══════════════════════════════════════════════════════════════

separator("LEVEL 8: Variational — Ansatz & QNG")

from qgt.variational.ansatz import AnsatzLibrary

lib = AnsatzLibrary(n_qubits=4)

# Xem các ansatz có sẵn
hea = lib.hardware_efficient(n_layers=3)
sel = lib.strongly_entangling(n_layers=2)
ucc = lib.uccsd_inspired(n_layers=1)
sym = lib.symmetry_preserving(n_layers=2)

print("Ansatz library:")
print(f"  {hea.name}: {hea.n_params} params")
print(f"  {sel.name}: {sel.n_params} params")
print(f"  {ucc.name}: {ucc.n_params} params")
print(f"  {sym.name}: {sym.n_params} params")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

separator("TÓM TẮT — Cách Sử Dụng Nhanh")

print("""
╔══════════════════════════════════════════════════════════════╗
║  CÁCH SỬ DỤNG QGT PACKAGE                                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. IMPORT MODEL:                                            ║
║     from qgt.models.ising import IsingModel                  ║
║     model = IsingModel(n_sites=6)                            ║
║                                                              ║
║  2. TÍNH QGT:                                                ║
║     from qgt.core.qgt import QuantumGeometricTensor          ║
║     qgt = QuantumGeometricTensor(model.ground_state, 2)      ║
║     g = qgt.metric(np.array([1.0, 0.5]))                    ║
║                                                              ║
║  3. GEOMETRY:                                                ║
║     from qgt.geometry.curvature import CurvatureAnalyzer     ║
║     curv = CurvatureAnalyzer(MetricTensor(qgt.metric))       ║
║     R = curv.scalar_curvature(params)                        ║
║                                                              ║
║  4. PHASE TRANSITION:                                        ║
║     from qgt.applications.phase_transitions import ...       ║
║     detector = PhaseTransitionDetector(model)                ║
║     result = detector.find_critical_points(...)              ║
║                                                              ║
║  5. VISUALIZE:                                               ║
║     from qgt.visualization.manifold_plots import ...         ║
║     plotter = ManifoldPlotter('output/')                     ║
║     plotter.plot_overview(qgt_scan)                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("✓ Demo hoàn tất!")
print(f"  Plots saved to: {output_dir}/")
