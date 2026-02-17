# Quantum Geometric Tensor (QGT): A Deep Dive

## 1. Introduction: Geometry of Quantum States

In quantum mechanics, a quantum state is represented by a vector $|\psi(\theta)\rangle$ in a Hilbert space, dependent on a set of parameters $\theta = (\theta_1, \theta_2, \dots, \theta_M)$.

The **Quantum Geometric Tensor (QGT)** is a fundamental object that describes the geometry of the parameter space induced by the quantum states. It tells us how the quantum state changes as we vary the parameters.

Intuitively:
*   **Real Part (Riemannian metric)**: Describes the "distance" or distinguishability between two nearby quantum states. This is the **Fubini-Study metric** ($g_{ij}$).
*   **Imaginary Part (Symplectic form)**: Describes the "phase" or curvature of the manifold. This is the **Berry curvature** ($\Omega_{ij}$).

## 2. Derivation Step-by-Step

We start by considering the distance between two infinitesimally close states $|\psi(\theta)\rangle$ and $|\psi(\theta + d\theta)\rangle$.

### 2.1. The Fidelity and Distance
The fidelity between two pure states is $F = |\langle \psi(\theta) | \psi(\theta + d\theta) \rangle|^2$.
The distance squared is often defined as $ds^2 = 1 - F$.

Expanding $|\psi(\theta + d\theta)\rangle$ to second order in $d\theta$:
$$
|\psi(\theta + d\theta)\rangle \approx |\psi(\theta)\rangle + \sum_i \frac{\partial |\psi\rangle}{\partial \theta_i} d\theta_i + \frac{1}{2} \sum_{i,j} \frac{\partial^2 |\psi\rangle}{\partial \theta_i \partial \theta_j} d\theta_i d\theta_j
$$

The overlap is:
$$
\langle \psi(\theta) | \psi(\theta + d\theta) \rangle \approx 1 + \sum_i \langle \psi | \partial_i \psi \rangle d\theta_i + \frac{1}{2} \sum_{i,j} \langle \psi | \partial_i \partial_j \psi \rangle d\theta_i d\theta_j
$$
where $\partial_i \equiv \frac{\partial}{\partial \theta_i}$.

Since $\langle \psi | \psi \rangle = 1$, differentiating gives $\langle \partial_i \psi | \psi \rangle + \langle \psi | \partial_i \psi \rangle = 0$.
So, $\text{Re}(\langle \psi | \partial_i \psi \rangle) = 0$. $\langle \psi | \partial_i \psi \rangle$ is purely imaginary (related to Berry connection).

After some algebra (keeping terms up to $d\theta^2$), the distance $ds^2$ becomes:
$$
ds^2 = \sum_{i,j} \left( \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle \right) d\theta_i d\theta_j
$$

### 2.2. Definition of the QGT
The term in the parenthesis is the Quantum Geometric Tensor $\chi_{ij}$:
$$
\chi_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle
$$

We can decompose it into real and imaginary parts:
$$
\chi_{ij} = g_{ij} + i \frac{\Omega_{ij}}{2}
$$

#### The Fubini-Study Metric Tensor ($g_{ij}$)
The symmetric, real part of QGT:
$$
g_{ij} = \text{Re}(\chi_{ij}) = \frac{1}{2} (\langle \partial_i \psi | \partial_j \psi \rangle + \langle \partial_j \psi | \partial_i \psi \rangle) - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle
$$
(Note: The second term assumes $\langle \psi | \partial_j \psi \rangle$ is purely imaginary, which is true for normalized states if we ignore global phase).

#### The Berry Curvature ($\Omega_{ij}$)
The antisymmetric, imaginary part of QGT:
$$
\Omega_{ij} = 2 \text{Im}(\chi_{ij}) = -i (\langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_j \psi | \partial_i \psi \rangle)
$$
This is related to the geometric phase accumulated during adiabatic evolution.

## 3. Why is this important?

1.  **Quantum Natural Gradient (QNG)**: Standard gradient descent follows the steepest path in parameter space (Euclidean geometry). But parameter space might be warped. QND uses $g_{ij}$ to follow the steepest path in *Hilbert space* (state space).
    $$ \theta_{new} = \theta - \eta g^{-1} \nabla L $$
2.  **Quantum Phase Transitions**: Singularities or discontinuities in QGT often signal quantum phase transitions.
3.  **Optimization Landscapes**: Understanding curvature ($\Omega_{ij}$) helps in diagnosing barren plateaus and local minima.

## 4. Application: Information Manifold of Quantum Field Theory

### 4.1. The Transverse-Field Ising Model as a Lattice QFT

The 1D Transverse-Field Ising Model (TFIM) is the simplest lattice quantum field theory exhibiting a quantum phase transition:

$$
H(J, h) = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i
$$

The parameter space $\mathcal{M} = \{(J, h) : J, h > 0\}$ forms a 2D manifold. The ground state $|\psi_0(J, h)\rangle$ defines a map from $\mathcal{M}$ into the projective Hilbert space, endowing it with the structure of a **Riemannian manifold** via the Fubini-Study metric.

### 4.2. Quantum Phase Transition as Geometric Singularity

At the critical point $h/J = 1$, the system undergoes a second-order quantum phase transition:
- **Ferromagnetic phase** ($h < J$): Spins align along $Z$, $\langle Z \rangle \neq 0$
- **Paramagnetic phase** ($h > J$): Spins align along $X$, $\langle Z \rangle = 0$

This transition manifests as a **geometric singularity** on the information manifold:
- The **fidelity susceptibility** $\chi_F = \mathrm{Tr}(g_{\mu\nu}) / N$ diverges as $|h - J|^{-\alpha}$
- The **metric determinant** $\det(g)$ peaks sharply along the critical line
- The **Ricci scalar curvature** $R$ exhibits a sign change or divergence
- **Geodesics** curve away from the critical line, reflecting the infinite "geometric distance" across the phase boundary

### 4.3. Differential Geometry on the Manifold

The full geometric analysis involves:

1. **Christoffel symbols**: $\Gamma^\sigma_{\mu\nu} = \frac{1}{2} g^{\sigma\rho}(\partial_\mu g_{\rho\nu} + \partial_\nu g_{\rho\mu} - \partial_\rho g_{\mu\nu})$
2. **Riemann curvature tensor**: $R^\sigma{}_{\rho\mu\nu} = \partial_\mu \Gamma^\sigma_{\nu\rho} - \partial_\nu \Gamma^\sigma_{\mu\rho} + \Gamma^\sigma_{\mu\lambda}\Gamma^\lambda_{\nu\rho} - \Gamma^\sigma_{\nu\lambda}\Gamma^\lambda_{\mu\rho}$
3. **Ricci scalar**: $R = g^{\rho\nu} R_{\rho\nu}$ where $R_{\rho\nu} = R^\mu{}_{\rho\mu\nu}$
4. **Geodesic equation**: $\ddot{x}^\sigma + \Gamma^\sigma_{\mu\nu} \dot{x}^\mu \dot{x}^\nu = 0$

### 4.4. Key Physical Insight

The information manifold framework reveals that quantum phase transitions are not just thermodynamic phenomena — they are **geometric singularities** in the space of quantum states. This connects condensed matter physics to information geometry and differential geometry in a profound way.
