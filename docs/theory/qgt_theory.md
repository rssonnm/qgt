# Quantum Geometric Tensor: Theory

## 1. Definition

The Quantum Geometric Tensor (QGT) for a parameterized family |ψ(λ)⟩:

$$\chi_{\mu\nu} = \langle \partial_\mu \psi | (1 - |\psi\rangle\langle\psi|) | \partial_\nu \psi \rangle$$

Equivalently:

$$\chi_{\mu\nu} = \langle \partial_\mu \psi | \partial_\nu \psi \rangle - \langle \partial_\mu \psi | \psi \rangle \langle \psi | \partial_\nu \psi \rangle$$

## 2. Decomposition

$$\chi_{\mu\nu} = g_{\mu\nu} + \frac{i}{2} \Omega_{\mu\nu}$$

- **g_μν = Re(χ)**: Fubini-Study metric — measures state distinguishability  
- **Ω_μν = 2·Im(χ)**: Berry curvature — encodes geometric phase

## 3. Key Properties

| Property | Metric g_μν | Berry Ω_μν |
|----------|-------------|------------|
| Symmetry | Symmetric | Antisymmetric |
| Type | Riemannian | Symplectic |
| Physical meaning | Distance | Phase |
| Phase transitions | Diverges at QPT | Sign change |

## 4. Derived Quantities

- **Fidelity susceptibility**: χ_F = Tr(g)/N (per site)
- **Quantum Fisher information**: F = 4g (Cramér-Rao bound)
- **Berry phase**: γ = ∮ A·dλ (Wilson loop)
- **Chern number**: C₁ = (1/2π) ∫∫ Ω dλ¹∧dλ² (topological)

## 5. Riemannian Geometry

From g_μν, full Riemannian geometry follows:
- Christoffel symbols Γ^σ_μν (connection)
- Riemann tensor R^σ_ρμν (curvature)
- Ricci scalar R (intrinsic curvature)
- Geodesic equation d²x^σ/dt² + Γ^σ_μν ẋ^μ ẋ^ν = 0

## 6. References

1. Provost & Vallee, CMP 76 (1980) — original QGT definition
2. Zanardi et al., PRL 99 (2007) — QGT and QPT
3. Stokes et al., Quantum 4 (2020) — QNG
