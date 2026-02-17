# Information Geometry of Quantum States

## Overview

The information manifold M = {|ψ(λ)⟩} is the set of all quantum states
reachable by varying the model parameters λ = (λ₁, ..., λ_n).

The QGT endows M with a Riemannian structure, making it a curved manifold.

## Ising Model Information Manifold

For the TFIM with parameters (J, h):
- The manifold is 2-dimensional
- The critical line h = J is a **geometric singularity**
- Geodesics deflect away from the critical line
- The Ricci scalar diverges at h/J = 1

### Physical Phases as Geometric Regions

| Phase | Region | Geometry |
|-------|--------|----------|
| Ferromagnetic | h < J | Large curvature near crit. |
| Paramagnetic | h > J | Flatter, lower curvature |
| Critical | h = J | Singularity (infinite curvature) |

### Finite-Size Scaling

For N sites, the fidelity susceptibility peak:
- Height: χ_F^peak ~ N^{2/ν} with ν = 1 (Ising)
- Position: h_c(N) → h_c^∞ = J as N → ∞

## Connection to Other Fields

- **Classical information geometry**: Fisher information → Cramér-Rao
- **Holographic duality (AdS/CFT)**: metric ↔ bulk geometry
- **Quantum error correction**: distance ↔ code distance
- **Machine learning**: natural gradient descent (Fisher-Rao)
