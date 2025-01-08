import numpy as np
import matplotlib.pyplot as plt
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




z_0_eq1 = np.array([[-np.pi/2+0.01], [np.pi/2-0.01], [-44]])
z_0_eq2 = np.array([[0], [0], [0]])
z_0_eq3 = np.array([[np.pi/2-0.01], [-np.pi/2+0.01], [+44]])
# z_0_eq1 = np.array([[np.pi+np.pi/2.1], [-np.pi/2.1], [-40]])
# z_0_eq2 = np.array([[np.pi-np.pi/2.1], [+np.pi/2.1], [+40]])
x_eq1, u_eq1 = newton_method(z_0_eq1)
x_eq2, u_eq2 = newton_method(z_0_eq2)
x_eq3, u_eq3 = newton_method(z_0_eq3)
t_0 = 4
t_1 = 0.7
t_2 = 0.8
t_3 = 0.7
t_4 = 4

# Initial state and input
print("Initial state:\t", x_eq1.T ,"\tInitial input:\t", u_eq1.T)
print("Final state:\t", x_eq3.T, "\tFinal Input:\t", u_eq3.T)

smooth_percentage = 1


# Parameters
r = 0.5  # Radius of the circular trajectory
omega = -2*np.pi/t_2  # Angular velocity (rad/s)
xc = 0  # Center x-coordinate
yc = -3 + r  # Center y-coordinate
# tf_circle = 1  # Total time for circular motion
L1 = 1.5  # Length of the first pendulum arm
L2 = 1.5  # Length of the second pendulum arm
dt = 1e-3  # Time step
K_eq = 44
theta1_sequence, theta2_sequence = generate_circular_trajectory(r, omega, xc, yc, L1, L2, t_2, dt)

x_reference_2 = np.zeros((4, len(theta1_sequence)))
u_reference_2 = np.zeros((1, len(theta1_sequence)))

x_reference_0, u_reference_0 = reference_trajectory.generate_trajectory(x_eq1, x_eq1, u_eq1, u_eq1, smooth_percentage, t_0)
x_reference_1, u_reference_1 = reference_trajectory.generate_trajectory(x_eq1, x_eq2, u_eq1, u_eq2, smooth_percentage, t_1)
x_reference_2[2, :] = theta1_sequence
x_reference_2[3, :] = theta2_sequence
u_reference_2[0, :] = K_eq * np.sin(x_reference_2[2,:])
x_reference_3, u_reference_3 = reference_trajectory.generate_trajectory(x_eq2, x_eq3, u_eq2, u_eq3, smooth_percentage, t_3)
x_reference_4, u_reference_4 = reference_trajectory.generate_trajectory(x_eq3, x_eq3, u_eq3, u_eq3, smooth_percentage, t_4)
x_reference = np.hstack((x_reference_0, x_reference_1, x_reference_2, x_reference_3, x_reference_4))
u_reference = np.hstack((u_reference_0, u_reference_1, u_reference_2, u_reference_3, u_reference_4))

# dm.save_mpc_trajectory(x_reference, u_reference)

# Define the path to the text file
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

anim(x_reference.T)
