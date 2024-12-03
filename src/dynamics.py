import numpy as np
import matplotlib.pyplot as plt

# Dynamics parameters
M1 = 2
M2 = 2
L1 = 1.5
L2 = 1.5
R1 = 0.75
R2 = 0.75
I1 = 1.5
I2 = 1.5
G = 9.81
F1 = 0.1
F2 = 0.1

F = np.array([[F1, 0],
              [ 0 ,F2]])

Z_2x2 = np.zeros((2,2))

def compute_inertia_matrix(theta2):
    """Compute the inertia matrix M."""
    cos_theta2 = np.cos(theta2)
    m11 = I1 + I2 + M1 * R1**2 + M2 * (L1**2 + R2**2) + 2 * M2 * L1 * R2 * cos_theta2
    m12 = I2 + M2 * R2**2 + M2 * L1 * R2 * cos_theta2
    m21 = m12
    m22 = I2 + M2 * R2**2
    return np.array([[m11, m12], [m21, m22]])

def compute_coriolis(theta2, dtheta1, dtheta2):
    """Compute the Coriolis and centrifugal forces matrix C."""
    sin_theta2 = np.sin(theta2)
    c1 = -M2 * L1 * R2 * dtheta2 * sin_theta2 * (dtheta2 + 2 * dtheta1)
    c2 = M2 * L1 * R2 * sin_theta2 * dtheta1**2
    return np.array([[c1], [c2]])

def compute_gravity(theta1, theta2):
    """Compute the gravity forces matrix G."""
    sin_theta1 = np.sin(theta1)
    sin_theta1_theta2 = np.sin(theta1 + theta2)
    g1 = G * (M1 * R1 + M2 * L1) * sin_theta1 + G * M2 * R2 * sin_theta1_theta2
    g2 = G * M2 * R2 * sin_theta1_theta2
    return np.array([[g1], [g2]])


def dynamics(x, u, dt=1e-3):

    theta1 = x[0].item()
    #print("theta1:",theta1)
    theta2 = x[1].item()
    #print("theta2:",theta2)
    dtheta1 = x[2].item()
    #print("dtheta1:",dtheta1)
    dtheta2 = x[3].item()
    #print("dtheta2:",dtheta2)

    # Compute matrices
    M = compute_inertia_matrix(theta2)
    M_inv = np.linalg.inv(M)
    C = compute_coriolis(theta2, dtheta1, dtheta2)
    G = compute_gravity(theta1, theta2)
    
    A = np.block([[ -M_inv @ F, Z_2x2 ], 
                  [ np.eye(2), Z_2x2 ]])    
    
    M_inv_ext = np.block([
        [M_inv, Z_2x2],
        [Z_2x2, Z_2x2]
    ])

    B = M_inv_ext
    
    C_ext = np.block([
        [C],
        [np.zeros((2, 1))]  # Ensure the zeros are in the same shape (2, 1)
    ])
    
    G_ext = np.block([
        [G],
        [np.zeros((2, 1))]
    ])
    
    
    # print("M:\n", M)
    # print("M_inv:\n", M_inv)
    # print("C:\n", C)
    # print("F:\n", F)
    # print("G:\n", G)
    # print("A:\n", A)
    # print("B:\n", B)
    # print("C_ext:\n", C_ext)
    # print("G_ext:\n", G_ext)
    
    # print("x:\n", x)
    # print("u:\n", u)
    
    # x_dot = A @ x + B @ u - M_inv_ext @ C_ext - M_inv_ext @ G_ext
    x_dot = A @ x + B @ u - M_inv_ext @ (C_ext + G_ext)
    # print("x1_dot:",x_dot[0])
    # print("x2_dot:",x_dot[1])
    # print("x3_dot:",x_dot[2])
    # print("x4_dot:",x_dot[3])
    
    x_new = x + dt * x_dot
    
    return x_new