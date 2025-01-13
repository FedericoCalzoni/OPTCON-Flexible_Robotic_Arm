import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from newton_method import newton_method
import reference_trajectory
from visualizer import animate_double_pendulum as anim
import data_manager as dm

np.set_printoptions(linewidth=100)

# Inverse Kinematics Function
def inverse_kinematics(x, y, L1, L2):
    """Calculate theta1 and theta2 for the given (x, y)."""
    # Calculate distance from the origin to the point
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if np.abs(D) > 1.0:
        raise ValueError("Target is out of reach for the double pendulum.")
    
    theta2 = np.arccos(D)  # Elbow angle
    phi = np.arctan2(x, y)
    k1 = L1 - L2 * np.cos(theta2)
    k2 = - L2 * np.sin(theta2)
    theta1 = phi + np.arctan2(k1, k2)
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    return theta1, theta2

# Function to Generate Circular Trajectory and Joint Angles
def generate_circular_trajectory(r, omega, xc, yc, L1, L2, tf_circle, dt):
    """Generate the trajectory and corresponding joint angles theta1 and theta2."""
    time_circle = np.arange(0, tf_circle, dt)
    
    # Circular trajectory
    x_d = r * np.cos(omega * time_circle - np.pi/2) + xc
    y_d = r * np.sin(omega * time_circle - np.pi/2) + yc
    
    # Initialize sequences
    theta1_sequence = []
    theta2_sequence = []
    
    # Inverse kinematics for each point
    for x, y in zip(x_d, y_d):
        theta1, theta2 = inverse_kinematics(x, y, L1, L2)
        theta1_sequence.append(theta1)
        theta2_sequence.append(theta2)
    
    return theta1_sequence, theta2_sequence


L1 = 1.5  # Length of the first pendulum arm
L2 = 1.5  # Length of the second pendulum arm
t_2 = 10
# Parameters
r = 0.2  # Radius of the circular trajectory
omega = -4*np.pi/t_2  # Angular velocity (rad/s)
xc = + L1 * np.cos(np.pi/4)  # Center x-coordinate
yc = - L1 * np.cos(np.pi/4) - L2# Center y-coordinate
# tf_circle = 1  # Total time for circular motion

dt = 1e-3  # Time step
K_eq = 44
theta1_sequence, theta2_sequence = generate_circular_trajectory(r, omega, xc, yc, L1, L2, t_2, dt)

x_reference = np.zeros((4, len(theta1_sequence)))
u_reference = np.zeros((1, len(theta1_sequence)))

x_reference[2, :] = theta1_sequence
x_reference[3, :] = theta2_sequence
u_reference[0, :] = K_eq * np.sin(x_reference[2,:])

# plot x_reference and u_reference
plt.figure(figsize=(10, 10))
plt.plot(x_reference[0, :], label=r'$\dot \theta_1^{ref}$', color='m', linewidth=2)
plt.plot(x_reference[1, :], label=r'$\dot \theta_2^{ref}$', color='orange', linewidth=2)
plt.plot(x_reference[2, :], label=r'$\theta_1^{ref}$', color='b', linewidth=2)
plt.plot(x_reference[3, :], label=r'$\theta_2^{ref}$', color='g', linewidth=2)
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(u_reference[0, :], label=r'$\tau_1^{ref}$', color='r', linewidth=2)
plt.legend()

print(x_reference.shape)

anim(x_reference.T)

dm.save_mpc_trajectory(x_reference, u_reference)
