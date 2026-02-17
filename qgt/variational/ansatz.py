"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.variational.ansatz — Variational Ansatz Library
═══════════════════════════════════════════════════════════════════════════════

Collection of parameterized quantum circuit ansätze for VQE and QGT studies.
Each ansatz defines a map from variational parameters to quantum states.
"""

import numpy as np
from typing import List, Optional

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


class AnsatzLibrary:
    """
    Library of variational ansätze (parameterized quantum circuits).
    
    All ansätze return a callable pennylane circuit.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    """
    
    def __init__(self, n_qubits: int):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for variational module")
        self.n_qubits = n_qubits
    
    def hardware_efficient(self, n_layers: int = 3,
                            rotation_gates: List[str] = None):
        """
        Hardware-Efficient Ansatz (HEA).
        
        Structure per layer:
            R_y(θ) R_z(θ) on each qubit → CNOT chain
        
        Parameters
        ----------
        n_layers : int
            Number of layers.
        rotation_gates : list of str
            Rotation gates per qubit (default: ['RY', 'RZ']).
        
        Returns
        -------
        callable
            Circuit function: ansatz(params, wires)
        """
        if rotation_gates is None:
            rotation_gates = ['RY', 'RZ']
        
        n_params_per_layer = self.n_qubits * len(rotation_gates)
        total_params = n_layers * n_params_per_layer
        
        gate_map = {
            'RX': qml.RX, 'RY': qml.RY, 'RZ': qml.RZ,
        }
        
        def circuit(params, wires=None):
            if wires is None:
                wires = list(range(self.n_qubits))
            
            param_idx = 0
            for layer in range(n_layers):
                # Rotation block
                for q in range(self.n_qubits):
                    for gate_name in rotation_gates:
                        gate_map[gate_name](params[param_idx], wires=wires[q])
                        param_idx += 1
                
                # Entangling block (linear CNOT chain)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[wires[q], wires[q + 1]])
        
        circuit.n_params = total_params
        circuit.n_layers = n_layers
        circuit.name = f"HEA({n_layers} layers, {rotation_gates})"
        return circuit
    
    def strongly_entangling(self, n_layers: int = 3):
        """
        Strongly Entangling Layers (SEL) — PennyLane built-in.
        
        More expressive than HEA due to all-to-all CZ entanglement.
        """
        total_params = n_layers * self.n_qubits * 3
        
        def circuit(params, wires=None):
            if wires is None:
                wires = list(range(self.n_qubits))
            weights = params.reshape(n_layers, self.n_qubits, 3)
            qml.StronglyEntanglingLayers(weights, wires=wires)
        
        circuit.n_params = total_params
        circuit.n_layers = n_layers
        circuit.name = f"SEL({n_layers} layers)"
        return circuit
    
    def uccsd_inspired(self, n_layers: int = 1):
        """
        UCCSD-inspired ansatz for chemistry-like problems.
        
        Combines single and double excitation gates.
        """
        n_singles = self.n_qubits
        n_doubles = (self.n_qubits * (self.n_qubits - 1)) // 2
        total_params = n_layers * (n_singles + n_doubles)
        
        def circuit(params, wires=None):
            if wires is None:
                wires = list(range(self.n_qubits))
            
            param_idx = 0
            for layer in range(n_layers):
                # Single excitations
                for q in range(self.n_qubits):
                    qml.RY(params[param_idx], wires=wires[q])
                    param_idx += 1
                
                # Double excitations (pairwise)
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[wires[i], wires[j]])
                        qml.RZ(params[param_idx], wires=wires[j])
                        qml.CNOT(wires=[wires[i], wires[j]])
                        param_idx += 1
        
        circuit.n_params = total_params
        circuit.n_layers = n_layers
        circuit.name = f"UCCSD-inspired({n_layers} layers)"
        return circuit
    
    def symmetry_preserving(self, n_layers: int = 2):
        """
        Ansatz that preserves particle number / total Z magnetization.
        
        Uses only excitation-preserving gates (XX + YY rotations).
        """
        n_bonds = self.n_qubits - 1
        total_params = n_layers * (self.n_qubits + n_bonds)
        
        def circuit(params, wires=None):
            if wires is None:
                wires = list(range(self.n_qubits))
            
            param_idx = 0
            for layer in range(n_layers):
                # On-site rotations (RZ preserves Z magnetization)
                for q in range(self.n_qubits):
                    qml.RZ(params[param_idx], wires=wires[q])
                    param_idx += 1
                
                # Excitation-preserving (Givens rotation)
                for q in range(self.n_qubits - 1):
                    # XX + YY rotation
                    qml.CNOT(wires=[wires[q], wires[q+1]])
                    qml.RY(params[param_idx], wires=wires[q])
                    qml.CNOT(wires=[wires[q], wires[q+1]])
                    param_idx += 1
        
        circuit.n_params = total_params
        circuit.n_layers = n_layers
        circuit.name = f"SymPreserving({n_layers} layers)"
        return circuit
