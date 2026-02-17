"""
Tests for qgt.geometry module.
"""

import numpy as np
import pytest
from qgt.geometry.metric import MetricTensor
from qgt.geometry.connection import ChristoffelSymbols
from qgt.geometry.curvature import CurvatureAnalyzer


class TestMetricTensor:
    @staticmethod
    def _flat_metric(params):
        """Euclidean metric (identity)."""
        return np.eye(2)
    
    @staticmethod
    def _sphere_metric(params):
        """Metric on a sphere: ds² = dθ² + sin²θ dφ²."""
        theta = params[0]
        return np.diag([1.0, np.sin(theta)**2 + 1e-12])
    
    def test_flat_determinant(self):
        m = MetricTensor(self._flat_metric)
        assert abs(m.determinant(np.array([0, 0])) - 1.0) < 1e-10
    
    def test_flat_condition_number(self):
        m = MetricTensor(self._flat_metric)
        assert abs(m.condition_number(np.array([0, 0])) - 1.0) < 1e-10
    
    def test_sphere_volume(self):
        m = MetricTensor(self._sphere_metric)
        vol = m.volume_element(np.array([np.pi/4, 0]))
        expected = np.sin(np.pi/4)
        assert abs(vol - expected) < 0.01


class TestChristoffel:
    def test_flat_space_vanishes(self):
        m = MetricTensor(lambda p: np.eye(2))
        chris = ChristoffelSymbols(m)
        Gamma = chris.compute(np.array([1.0, 1.0]))
        assert np.max(np.abs(Gamma)) < 0.01


class TestCurvature:
    def test_flat_curvature_zero(self):
        m = MetricTensor(lambda p: np.eye(2))
        curv = CurvatureAnalyzer(m)
        R = curv.scalar_curvature(np.array([1.0, 1.0]))
        assert abs(R) < 0.1  # numerical noise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
