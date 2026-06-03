import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgt.models.ising import IsingModel
from qgt.core.qgt import QuantumGeometricTensor
from qgt.core.states import QuantumState
from qgt.visualization.styles import set_academic_style, COLORS


def main():
    set_academic_style()
    outdir = 'output/examples'
    os.makedirs(outdir, exist_ok=True)

    N = 6
    model = IsingModel(n_sites=N, periodic=True)
    qgt = QuantumGeometricTensor(model.ground_state, n_params=2)
    subsystem = list(range(N // 2))

    print(f"── Entanglement ↔ Geometry (N={N}) ─────────────")
    print(f"   Subsystem A = sites {subsystem}")

    h_vals = np.linspace(0.1, 2.0, 40)
    S_ent = np.zeros(len(h_vals))
    chi_F = np.zeros(len(h_vals))

    for i, h in enumerate(h_vals):
        params = np.array([1.0, h])
        _, psi = model.ground_state(params)

        # Entanglement entropy
        qs = QuantumState(psi, normalize=False)
        S_ent[i] = qs.entanglement_entropy(subsystem, n_qubits=N)

        # Fidelity susceptibility
        g = qgt.metric(params)
        chi_F[i] = np.trace(g) / N

    # Hệ số tương quan Pearson
    corr = np.corrcoef(S_ent, chi_F)[0, 1]
    print(f"\n  Pearson correlation: r = {corr:.4f}")
    print(f"  → {'Tương quan mạnh' if abs(corr) > 0.7 else 'Tương quan trung bình'}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(h_vals, S_ent, color=COLORS['accent'], linewidth=2)
    axes[0].axvline(1.0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('h / J')
    axes[0].set_ylabel('S(ρ_A)')
    axes[0].set_title('Entanglement Entropy')

    axes[1].plot(h_vals, chi_F, color=COLORS['secondary'], linewidth=2)
    axes[1].axvline(1.0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('h / J')
    axes[1].set_ylabel('χ_F')
    axes[1].set_title('Fidelity Susceptibility')

    axes[2].scatter(S_ent, chi_F, c=h_vals, cmap='coolwarm',
                     s=40, alpha=0.8)
    axes[2].set_xlabel('S(ρ_A)')
    axes[2].set_ylabel('χ_F')
    axes[2].set_title(f'Correlation: r = {corr:.3f}')

    z = np.polyfit(S_ent, chi_F, 1)
    x_fit = np.linspace(S_ent.min(), S_ent.max(), 50)
    axes[2].plot(x_fit, np.polyval(z, x_fit), 'k--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(f'{outdir}/entanglement_geometry_link.png',
                 dpi=200, bbox_inches='tight')
    print(f"\nPlot: {outdir}/entanglement_geometry_link.png")


if __name__ == "__main__":
    main()
