"""
Tests for qgt.variational module (requires PennyLane).
"""

import numpy as np
import pytest

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
class TestAnsatzLibrary:
    def test_hea_param_count(self):
        from qgt.variational.ansatz import AnsatzLibrary
        lib = AnsatzLibrary(n_qubits=4)
        hea = lib.hardware_efficient(n_layers=3)
        # 4 qubits × 2 gates × 3 layers = 24
        assert hea.n_params == 24
    
    def test_hea_runs(self):
        from qgt.variational.ansatz import AnsatzLibrary
        lib = AnsatzLibrary(n_qubits=2)
        hea = lib.hardware_efficient(n_layers=2)
        
        dev = qml.device('default.qubit', wires=2)
        
        @qml.qnode(dev)
        def circuit(params):
            hea(params)
            return qml.state()
        
        params = np.random.randn(hea.n_params)
        state = circuit(params)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
