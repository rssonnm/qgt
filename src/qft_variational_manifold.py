"""
═══════════════════════════════════════════════════════════════════════════════
  QGT Variational Extension: PennyLane VQE for Information Manifold
═══════════════════════════════════════════════════════════════════════════════

  This module uses PennyLane to build a VARIATIONAL QUANTUM EIGENSOLVER (VQE)
  that approximates the ground state of the Transverse-Field Ising Model.

  We then compute the QGT on BOTH:
    1. The "physical" manifold (parameterized by H(J, h))
    2. The "variational" manifold (parameterized by circuit angles θ)

  Key Question:
  ─────────────
  Does the variational circuit faithfully reproduce the geometry
  of the true ground state manifold?

  If yes → the variational ansatz has enough EXPRESSIVITY.
  If no  → the ansatz is too restricted (can't capture the full geometry).

  This is a powerful diagnostic for ansatz design!

  Author: WSQuantumAI
  Optimized for: Apple Silicon M4
═══════════════════════════════════════════════════════════════════════════════
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: DEFINE THE ISING HAMILTONIAN IN PENNYLANE
# ═══════════════════════════════════════════════════════════════════════════

def ising_hamiltonian(J, h, n_qubits=4, periodic=True):
    """
    Build the TFIM Hamiltonian as a PennyLane Hamiltonian object.
    
    H = -J Σ Z_i Z_{i+1} - h Σ X_i
    """
    coeffs = []
    obs = []
    
    # ZZ interactions
    n_bonds = n_qubits if periodic else n_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % n_qubits
        coeffs.append(-J)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    # Transverse field
    for i in range(n_qubits):
        coeffs.append(-h)
        obs.append(qml.PauliX(i))
    
    return qml.Hamiltonian(coeffs, obs)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: VARIATIONAL ANSATZ
# ═══════════════════════════════════════════════════════════════════════════

N_QUBITS = 4
N_LAYERS = 3

# Device: default.qubit for backprop-based gradient (fast on M4)
dev_vqe = qml.device("default.qubit", wires=N_QUBITS)
# Device with auxiliary wire for metric tensor
dev_mt = qml.device("lightning.qubit", wires=N_QUBITS + 1)


def ansatz(params, wires):
    """
    Hardware-Efficient Ansatz (HEA) for the TFIM ground state.
    
    Structure (per layer):
        1. Ry rotation on each qubit
        2. Rz rotation on each qubit
        3. CNOT entanglement (nearest-neighbor)
    
    Total parameters: n_layers × n_qubits × 2
    """
    n_qubits = len(wires)
    n_layers = params.shape[0]
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            qml.RY(params[layer, q, 0], wires=wires[q])
            qml.RZ(params[layer, q, 1], wires=wires[q])
        
        # Entanglement
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[wires[q], wires[q + 1]])
        if n_qubits > 1:
            qml.CNOT(wires=[wires[n_qubits - 1], wires[0]])


@qml.qnode(dev_vqe, diff_method="backprop")
def vqe_cost(params, J, h):
    """VQE cost function: ⟨ψ(θ)|H(J,h)|ψ(θ)⟩"""
    ansatz(params, wires=range(N_QUBITS))
    H = ising_hamiltonian(J, h, N_QUBITS)
    return qml.expval(H)


@qml.qnode(dev_vqe, diff_method="backprop")
def vqe_state(params):
    """Returns the state vector of the variational circuit."""
    ansatz(params, wires=range(N_QUBITS))
    return qml.state()


# For metric tensor calculation
@qml.qnode(dev_mt)
def circuit_for_mt(params_flat):
    """Flattened-parameter circuit for metric_tensor transform."""
    p = params_flat.reshape(N_LAYERS, N_QUBITS, 2)
    ansatz(p, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: VQE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def optimize_vqe(J, h, n_steps=100, lr=0.1):
    """
    Run VQE to find the ground state of H(J, h).
    
    Returns optimized parameters.
    """
    np.random.seed(42)
    params = np.random.uniform(-np.pi, np.pi, 
                                size=(N_LAYERS, N_QUBITS, 2),
                                requires_grad=True)
    
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    
    for step in range(n_steps):
        params = opt.step(lambda p: vqe_cost(p, J, h), params)
    
    final_energy = vqe_cost(params, J, h)
    return params, float(final_energy)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: QGT ON THE VARIATIONAL MANIFOLD
# ═══════════════════════════════════════════════════════════════════════════

def compute_variational_qgt(params):
    """
    Compute the full QGT on the variational parameter space
    using PennyLane's state vector + Jacobian approach.
    
    Returns the full (n_params × n_params) QGT matrix.
    """
    params_flat = params.flatten()
    psi = vqe_state(params)
    
    # Compute Jacobian of the state w.r.t. flattened params
    def state_real_imag(p_flat):
        p_reshaped = p_flat.reshape(N_LAYERS, N_QUBITS, 2)
        s = vqe_state(p_reshaped)
        return np.concatenate([np.real(s), np.imag(s)])
    
    jac_fn = qml.jacobian(state_real_imag)
    J_ri = jac_fn(params_flat)
    
    # Reconstruct complex Jacobian
    dim = len(psi)
    J_real = J_ri[:dim, :]
    J_imag = J_ri[dim:, :]
    J_mat = J_real + 1j * J_imag
    
    # QGT = J^† J - (J^† |ψ⟩)(⟨ψ| J)
    term1 = np.conj(J_mat.T) @ J_mat
    v = np.conj(J_mat.T) @ psi
    term2 = np.outer(v, np.conj(v))
    
    qgt = term1 - term2
    return np.real(qgt)  # Return Fubini-Study metric (real part)


def compute_fidelity_susceptibility_variational(J_val, h_range, n_points=20):
    """
    For each h, optimize VQE, then compute the fidelity susceptibility
    as Tr(g) where g is the variational metric tensor.
    """
    h_vals = np.linspace(h_range[0], h_range[1], n_points)
    chi_F = np.zeros(n_points)
    energies = np.zeros(n_points)
    
    for i, h in enumerate(h_vals):
        print(f"    VQE at h={h:.2f} ({i+1}/{n_points})", end='\r')
        params, E = optimize_vqe(J_val, h, n_steps=80, lr=0.15)
        energies[i] = E
        
        try:
            g = compute_variational_qgt(params)
            chi_F[i] = np.trace(g) / N_QUBITS
        except Exception:
            chi_F[i] = 0.0
    
    print()
    return h_vals, chi_F, energies


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: COMPARISON: EXACT vs VARIATIONAL
# ═══════════════════════════════════════════════════════════════════════════

def run_comparison():
    """
    Compare the fidelity susceptibility from:
      - Exact diagonalization (from qft_information_manifold.py)
      - Variational QGT (this module)
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   VARIATIONAL QGT: Information Manifold via VQE               ║
    ║                                                                ║
    ║   Comparing exact vs variational geometry                     ║
    ║                                                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    J_val = 1.0
    h_range = (0.2, 1.8)
    n_pts = 15
    
    t0 = time.time()
    
    # ── Variational computation ──
    print("  [Variational] Computing fidelity susceptibility via VQE...")
    h_var, chi_var, E_var = compute_fidelity_susceptibility_variational(
        J_val, h_range, n_points=n_pts
    )
    
    # ── Exact computation (import from main module) ──
    print("  [Exact] Computing fidelity susceptibility via exact diag...")
    try:
        from qft_information_manifold import IsingFieldTheory, QGTEngine
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from qft_information_manifold import IsingFieldTheory, QGTEngine
    
    ft = IsingFieldTheory(n_sites=N_QUBITS)
    engine = QGTEngine(ft)
    
    chi_exact = np.zeros(n_pts)
    E_exact = np.zeros(n_pts)
    for i, h in enumerate(h_var):
        g = engine.compute_metric(J_val, h)
        chi_exact[i] = np.trace(g) / N_QUBITS
        E, _ = ft.ground_state(J_val, h)
        E_exact[i] = E
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    
    # ── Visualization ──
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Exact vs Variational Information Geometry\n'
        f'{N_QUBITS}-qubit TFIM at J={J_val}',
        fontsize=16, fontweight='bold'
    )
    
    # Panel (a): Fidelity susceptibility
    axes[0].plot(h_var, chi_exact, 'b-o', lw=2, ms=5, label='Exact (ED)')
    axes[0].plot(h_var, chi_var, 'r--s', lw=2, ms=5, label=f'Variational ({N_LAYERS}-layer HEA)')
    axes[0].axvline(x=J_val, color='gray', ls=':', alpha=0.7, label=r'$h_c = J$')
    axes[0].set_xlabel(r'Transverse field $h$', fontsize=13)
    axes[0].set_ylabel(r'$\chi_F = \mathrm{Tr}(g) / N$', fontsize=13)
    axes[0].set_title('(a) Fidelity Susceptibility', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Panel (b): Ground state energy
    axes[1].plot(h_var, E_exact, 'b-o', lw=2, ms=5, label='Exact (ED)')
    axes[1].plot(h_var, E_var, 'r--s', lw=2, ms=5, label=f'VQE ({N_LAYERS}-layer HEA)')
    axes[1].axvline(x=J_val, color='gray', ls=':', alpha=0.7, label=r'$h_c = J$')
    axes[1].set_xlabel(r'Transverse field $h$', fontsize=13)
    axes[1].set_ylabel(r'Ground state energy $E_0$', fontsize=13)
    axes[1].set_title('(b) Energy Comparison', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exact_vs_variational_qgt.png',
                bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  → Saved: {output_dir}/exact_vs_variational_qgt.png")
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  ANALYSIS")
    print(f"{'='*60}")
    energy_error = np.mean(np.abs(E_var - E_exact))
    print(f"  Mean |E_VQE - E_exact|: {energy_error:.6f}")
    
    # Check if the variational susceptibility peaks near h_c
    h_peak_exact = h_var[np.argmax(chi_exact)]
    h_peak_var   = h_var[np.argmax(chi_var)]
    print(f"  Peak χ_F (exact):  h = {h_peak_exact:.2f}")
    print(f"  Peak χ_F (VQE):    h = {h_peak_var:.2f}")
    
    if abs(h_peak_exact - h_peak_var) < 0.3:
        print(f"\n  ✓ The variational ansatz CAPTURES the phase transition geometry!")
    else:
        print(f"\n  ✗ The variational ansatz MISSES the transition → needs more layers.")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run_comparison()
