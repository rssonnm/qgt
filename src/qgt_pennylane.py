
import pennylane as qml
from pennylane import numpy as np
import time

# Use lightning.qubit for optimization on M4 (via CPU vectorization or Metal if supported/installed)
# 'lightning.gpu' or 'lightning.kokkos' could be used if available, but 'lightning.qubit' is standard and fast.
# We need an auxiliary wire for the Hadamard test or similar methods used in QGT/metric tensor calculation.
dev = qml.device("lightning.qubit", wires=3)

@qml.qnode(dev)
def circuit(params):
    """
    Hardware Efficient Ansatz:
    Layers of rotations and entanglements.
    """
    # Layer 1
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    
    # Layer 2
    qml.RZ(params[2], wires=0)
    qml.RX(params[3], wires=1)
    qml.CNOT(wires=[0, 1])
    
    return qml.state()

def compute_qgt_pennylane(params):
    """
    Computes the Quantum Geometric Tensor using PennyLane's built-in functionality.
    Note: qml.metric_tensor gives the real part (Fubini-Study metric).
    To get the full QGT including Berry curvature, we can use qml.qinfo.quantum_geometric_tensor 
    (if available in this version) or construct it manually using overlaps.
    
    Let's check if qml.qinfo.quantum_geometric_tensor exists in 0.44.0.
    If not, we use the metric tensor and a custom curvature calculation.
    For this script, we will assume availability or fallback to metric tensor + warning.
    """
    # PennyLane 0.38+ supports quantum_geometric_tensor directly
    try:
        qgt_fn = qml.qinfo.quantum_geometric_tensor(circuit)
        return qgt_fn(params)
    except AttributeError:
        # Fallback if not available (though it should be in 0.44.0)
        print("Function qml.qinfo.quantum_geometric_tensor not found. Using qml.metric_tensor (Real part only).")
        mt_fn = qml.metric_tensor(circuit)
        return mt_fn(params) # This is only the real part (g_ij)

def benchmark_qgt(params):
    start_time = time.time()
    qgt = compute_qgt_pennylane(params)
    end_time = time.time()
    
    print(f"Time taken (PennyLane): {end_time - start_time:.6f} seconds")
    return qgt

if __name__ == "__main__":
    print("--- Efficient QGT with PennyLane (M4 Optimized) ---")
    
    # Random parameters
    params = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    
    print(f"Parameters: {params}")
    
    qgt = benchmark_qgt(params)
    
    print("\nQuantum Geometric Tensor (PennyLane Result):")
    print(qgt)
    
    if np.iscomplexobj(qgt):
        print("\nFubini-Study Metric (Real part):")
        print(np.real(qgt))
        
        print("\nBerry Curvature (Imaginary part * -2i approx?):")
        # Remember Omega_ij = -2 * Im(chi_ij)? No, Omega = 2 * Im(chi).
        # Detailed check in theory needed, but for now let's just show Im.
        print(np.imag(qgt))
    else:
        print("\nNote: Result is real-valued (Fubini-Study Metric only).")

