# Quantum Geometric Tensor

A comprehensive Python framework for studying the **geometry of quantum state spaces**, from fundamentals to advanced research applications.

## Architecture

```
qgt/
├── core/           # QGT, Fubini-Study metric, states, operators
├── geometry/       # Metric, Christoffel, curvature, geodesics, topology
├── models/         # Ising, Heisenberg, J1-J2 (+ base class for new models)
├── variational/    # Ansatz library, VQE, QNG optimizer
├── applications/   # Phase transitions, information manifold, entanglement
├── visualization/  # Publication-quality plotting
└── utils/          # Linear algebra, finite differences, logging
```

## Quick Start

```python
import numpy as np
from qgt.models.ising import IsingModel
from qgt.core.qgt import QuantumGeometricTensor

# Create model and QGT engine
model = IsingModel(n_sites=4)
qgt = QuantumGeometricTensor(model.ground_state, n_params=2)

# Compute QGT at a point
chi = qgt.compute(np.array([1.0, 0.5]))  # (J, h)
g = qgt.metric(np.array([1.0, 0.5]))     # Fubini-Study metric
omega = qgt.berry_curvature(np.array([1.0, 0.5]))  # Berry curvature
```

## Installation

```bash
# Core (numpy + scipy + matplotlib)
pip install -e .

# With PennyLane for variational
pip install -e ".[variational]"

# Development
pip install -e ".[dev]"
```

## Running

```bash
# Tests
python -m pytest tests/ -v

# Ising information manifold
python scripts/run_ising_manifold.py --n_sites 4 --n_points 25

# Benchmarks
python scripts/benchmark.py
```

## Key Features

| Module | Features |
|--------|----------|
| **core** | QGT, Fubini-Study metric, Berry curvature, Berry phase, fidelity susceptibility |
| **geometry** | Christoffel symbols, Riemann tensor, Ricci scalar, geodesics, Chern number |
| **models** | Ising (TFIM), Heisenberg (XXZ), J1-J2 SU(2), abstract base class |
| **variational** | HEA/SEL/UCCSD ansätze, VQE with QGT tracking, QNG optimizer |
| **applications** | QPT detection, finite-size scaling, entanglement-geometry correlation |

## Adding a New Model

```python
from qgt.models.base import QuantumModel

class MyModel(QuantumModel):
    def __init__(self, n_sites):
        super().__init__(name="My Model", param_names=["J", "h"], n_sites=n_sites)
    
    def hamiltonian(self, params):
        # Build your Hamiltonian
        ...
    
    def ground_state(self, params):
        H = self.hamiltonian(params)
        E, V = eigsh(H, k=1, which='SA')
        return float(E[0]), V[:, 0]
```

## Dependencies

- **Core**: numpy, scipy, matplotlib
- **Variational** (optional): pennylane
- **Dev**: pytest, black, isort

## License

MIT
