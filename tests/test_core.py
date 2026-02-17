"""
Tests for qgt.core module — QGT, Fubini-Study, states, operators.
"""

import numpy as np
import pytest
from qgt.core.states import QuantumState
from qgt.core.operators import PauliOperators
from qgt.core.qgt import QuantumGeometricTensor
from qgt.core.fubini_study import FubiniStudyMetric


class TestQuantumState:
    def test_normalization(self):
        psi = QuantumState([1, 0, 0, 1])
        assert abs(psi.norm() - 1.0) < 1e-10
    
    def test_from_label(self):
        psi = QuantumState.from_label('01')
        assert abs(psi.vector[1]) > 0.99
    
    def test_fidelity_same(self):
        psi = QuantumState.random(2, seed=42)
        assert abs(psi.fidelity(psi) - 1.0) < 1e-10
    
    def test_fidelity_orthogonal(self):
        psi = QuantumState.from_label('0')
        phi = QuantumState.from_label('1')
        assert psi.fidelity(phi) < 1e-10
    
    def test_probabilities(self):
        psi = QuantumState([1, 1], normalize=True)
        probs = psi.probabilities()
        np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-10)
    
    def test_entropy(self):
        psi = QuantumState([1, 1], normalize=True)
        assert abs(psi.entropy() - 1.0) < 1e-10  # max entropy for 2D
    
    def test_density_matrix(self):
        psi = QuantumState.from_label('0')
        rho = psi.density_matrix()
        assert abs(rho[0, 0] - 1.0) < 1e-10
        assert abs(np.trace(rho) - 1.0) < 1e-10


class TestPauliOperators:
    def test_commutation_XY(self):
        p = PauliOperators()
        comm = p.X @ p.Y - p.Y @ p.X  # = 2i Z
        expected = 2j * p.Z
        np.testing.assert_allclose(comm, expected, atol=1e-10)
    
    def test_anticommutation(self):
        p = PauliOperators()
        anticomm = p.X @ p.Y + p.Y @ p.X
        np.testing.assert_allclose(anticomm, np.zeros((2, 2)), atol=1e-10)
    
    def test_eigenvalues(self):
        p = PauliOperators()
        evals = np.linalg.eigvalsh(p.Z)
        np.testing.assert_allclose(sorted(evals), [-1, 1], atol=1e-10)
    
    def test_n_body_dimension(self):
        p = PauliOperators()
        op = p.n_body_operator({0: p.Z}, n_sites=4)
        assert op.shape == (16, 16)


class TestQGT:
    @staticmethod
    def _simple_state_fn(params):
        """Single-qubit rotation state |ψ(θ)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩."""
        theta = params[0]
        psi = np.array([np.cos(theta/2), np.sin(theta/2)])
        return 0.0, psi
    
    def test_metric_single_qubit(self):
        """For a single qubit rotation, g = 1/4."""
        qgt = QuantumGeometricTensor(self._simple_state_fn, n_params=1)
        g = qgt.metric(np.array([1.0]))
        assert abs(g[0, 0] - 0.25) < 0.01
    
    def test_metric_positive_semidefinite(self):
        qgt = QuantumGeometricTensor(self._simple_state_fn, n_params=1)
        g = qgt.metric(np.array([0.5]))
        assert g[0, 0] >= -1e-10
    
    def test_berry_curvature_real_state(self):
        """Berry curvature vanishes for real-valued states."""
        qgt = QuantumGeometricTensor(self._simple_state_fn, n_params=1)
        omega = qgt.berry_curvature(np.array([1.0]))
        assert abs(omega[0, 0]) < 1e-8


class TestFubiniStudy:
    def test_fidelity(self):
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([1, 0], dtype=complex)
        assert abs(FubiniStudyMetric.fidelity(psi, phi) - 1.0) < 1e-10
    
    def test_bures_same_state(self):
        psi = np.array([1, 0], dtype=complex)
        assert abs(FubiniStudyMetric.bures_distance(psi, psi)) < 1e-10
    
    def test_fs_distance_orthogonal(self):
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([0, 1], dtype=complex)
        d = FubiniStudyMetric.fubini_study_distance(psi, phi)
        assert abs(d - np.pi/2) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
