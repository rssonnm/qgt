import numpy as np

def rotation_matrix_y(theta):
    """Returns the rotation matrix for Ry(theta)."""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ])

def rotation_matrix_z(phi):
    """Returns the rotation matrix for Rz(phi)."""
    return np.array([
        [np.exp(-1j * phi / 2), 0],
        [0, np.exp(1j * phi / 2)]
    ])

def simple_ansatz(params):
    """
    A simple 1-qubit ansatz: |psi(theta, phi)> = Rz(phi) Ry(theta) |0>
    params: [theta, phi]
    """
    theta, phi = params
    state_0 = np.array([1, 0])
    
    # Apply Ry(theta)
    state = rotation_matrix_y(theta) @ state_0
    
    # Apply Rz(phi)
    state = rotation_matrix_z(phi) @ state
    
    return state

def compute_derivatives_numerical(params, epsilon=1e-5):
    """
    Computes finite-difference derivatives of the state vector w.r.t params.
    """
    grads = []
    psi_0 = simple_ansatz(params)
    
    for i in range(len(params)):
        params_shifted = params.copy()
        params_shifted[i] += epsilon
        psi_shifted = simple_ansatz(params_shifted)
        
        # Finite difference: (psi(theta+eps) - psi(theta)) / eps
        deriv = (psi_shifted - psi_0) / epsilon
        grads.append(deriv)
        
    return grads, psi_0

def compute_qgt_scratch(params):
    """
    Computes the Quantum Geometric Tensor (QGT) from scratch.
    QGT_ij = <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi>
    """
    grads, psi = compute_derivatives_numerical(params)
    n_params = len(params)
    qgt = np.zeros((n_params, n_params), dtype=complex)
    
    for i in range(n_params):
        for j in range(n_params):
            # Term 1: <d_i psi | d_j psi>
            term1 = np.vdot(grads[i], grads[j])
            
            # Term 2: <d_i psi | psi> <psi | d_j psi>
            term2a = np.vdot(grads[i], psi)
            term2b = np.vdot(psi, grads[j])
            term2 = term2a * term2b
            
            qgt[i, j] = term1 - term2
            
    return qgt

def main():
    print("--- QGT from Scratch (Educational) ---")
    # Parameters for the qubit [theta, phi]
    params = np.array([np.pi/2, np.pi/4])
    print(f"Parameters: theta={params[0]:.4f}, phi={params[1]:.4f}")
    
    qgt = compute_qgt_scratch(params)
    
    # Analytical Result for this ansatz (Bloch sphere):
    # Metric g = diag(1/4, sin^2(theta)/4)
    # Curvature Omega = 
    # Let's just print the calculated values.
    
    print("\nQuantum Geometric Tensor (chi_ij):")
    print(qgt)
    
    g_ij = np.real(qgt)
    omega_ij = 2 * np.imag(qgt)
    
    print("\nFubini-Study Metric (g_ij = Re(chi)):")
    print(g_ij)
    
    print("\nBerry Curvature (Omega_ij = 2*Im(chi)):")
    print(omega_ij)
    
    print("\nInterpretation:")
    print("g_ij tells us how 'fast' the state changes in different distinct directions.")
    print("Omega_ij relates to the geometric phase accumulated.")

if __name__ == "__main__":
    main()
