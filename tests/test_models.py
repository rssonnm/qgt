"""
Tests for qgt.models module.
"""

import numpy as np
import pytest
from qgt.models.ising import IsingModel
from qgt.models.heisenberg import HeisenbergModel


class TestIsingModel:
    def test_ground_state_energy(self):
        """Ground state energy of ferromagnetic Ising at h=0 should be -J*N."""
        model = IsingModel(n_sites=4, periodic=True)
        E, _ = model.ground_state(np.array([1.0, 0.0001]))
        assert E < -3.5  # Should be close to -4 for J=1, h~0
    
    def test_paramagnetic_energy(self):
        """At J=0, trivial system: E = -h*N."""
        model = IsingModel(n_sites=4, periodic=True)
        E, _ = model.ground_state(np.array([0.0001, 1.0]))
        assert E < -3.5  # Should be close to -4 for h=1, J~0
    
    def test_energy_gap_closes_at_critical(self):
        """Gap should be small near h=J=1 vs deep paramagnetic (h >> J)."""
        model = IsingModel(n_sites=6, periodic=True)
        gap_crit = model.energy_gap(np.array([1.0, 1.0]))
        gap_off = model.energy_gap(np.array([1.0, 2.0]))
        assert gap_crit < gap_off
    
    def test_hamiltonian_hermitian(self):
        model = IsingModel(n_sites=3)
        H = model.hamiltonian(np.array([1.0, 0.5]))
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.T, atol=1e-10)
    
    def test_state_normalized(self):
        model = IsingModel(n_sites=4)
        _, psi = model.ground_state(np.array([1.0, 0.5]))
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


class TestHeisenbergModel:
    def test_hamiltonian_hermitian(self):
        model = HeisenbergModel(n_sites=3)
        H = model.hamiltonian(np.array([1.0, 1.0, 0.0]))
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-10)
    
    def test_isotropic_point(self):
        """At the isotropic point (J=1, Δ=1), should give finite energy."""
        model = HeisenbergModel(n_sites=4)
        E, psi = model.ground_state(np.array([1.0, 1.0, 0.0]))
        assert E < 0  # Antiferro ground state has negative energy
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
