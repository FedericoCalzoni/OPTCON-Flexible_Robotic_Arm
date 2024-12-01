import numpy as np
import matplotlib.pyplot as plt

# Constants
DT = 1e-3

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
    print(m11)
    print(m12)
    print(m21)
    print(m22)
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


def dynamics(x, u):

    theta1 = x[0].item()
    theta2 = x[1].item()
    dtheta1 = x[2].item()
    dtheta2 = x[3].item()

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
    
    # dx = A @ x + B @ u - M_inv_ext @ C_ext - M_inv_ext @ G_ext
    dx = A @ x + B @ u - M_inv_ext @ (C_ext + G_ext)
    
    # print("dx:\n", dx)
    
    x_new = x + DT * dx
    
    return x_new

def plot_double_pendulum(theta1, theta2, l1=1.5, l2=1.5):
    # Calculate the positions of the pendulum arms
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # Update the plot of the double pendulum
    plt.clf()
    plt.plot([0, x1], [0, y1], 'o-', lw=2)
    plt.plot([x1, x2], [y1, y2], 'o-', lw=2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Double Pendulum')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()

def main():
    """Main function to simulate the dynamics."""
    # Initial state and input
    x = np.array([[3], [1.5], [1], [1]])  # Column vector for [theta1, theta2, dtheta1, dtheta2]
    u = np.array([[0], [0], [0], [0]])  # Column vector for [tau1, tau2]

    
    # print("x:\n", x)
    # print("u:\n", u)
    
    iteration = 0
    while True:
        x = dynamics(x, u)
        print("x:\n", x)
        if iteration % 10 == 0:
            plot_double_pendulum(x[0].item(), x[1].item())
        iteration += 1
    
    
if __name__ == "__main__":
    main()
