import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import parameters as pm 
from visualizer import animate_double_pendulum as anim
import data_manager as dm


# Inverse Kinematics Function
def inverse_kinematics(x, y, L1=pm.L1, L2=pm.L2, configuration=1):
    """Calculate theta1 and theta2 for the given (x, y)."""
    x = -x
    # Calculate distance from the origin to the point
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if np.abs(D) > 1:
        raise ValueError("Target is out of reach for the double pendulum.")
    
    theta2 = np.arccos(D) 
    phi = np.arctan2(x, y)
    k1 = L1 - L2 * np.cos(theta2)
    k2 = - L2 * np.sin(theta2)
    
    if configuration == 1:     
        theta1 = phi + np.arctan2(k1, k2)
    
    if configuration == 2:
        theta1 = phi - np.arctan2(k1, k2)
        theta2 = -theta2
        
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    
    
    # # if theta 1 and theta 2 are both very close to zero, set them to zero
    # if np.abs(theta1) < 0.0004  and np.abs(theta2) < 0.00009:
    #     print("theta1 and theta2 are zero")
    #     theta1 = 0
    #     theta2 = 0
    
    return theta1, theta2


def generate_circular_trajectory(r, omega, xc, yc, tf_circle, num_points=100):
    t = np.linspace(0, tf_circle, num_points)
    theta = omega * t
    x = xc + r * np.sin(theta+np.pi)
    y = yc + r * np.cos(theta+np.pi)
    return x.tolist(), y.tolist()

def generate_spline():
    radius = 3  # Circle radius
    
    # Define the start and end angles
    start_angle = np.pi / 5
    end_angle = np.pi / 16
    big_radius = 2.24
    
    # Generate the angles from np.pi/6 to 12
    angles = np.linspace(start_angle, end_angle, num=1)
    x_center = 0
    y_center = -0.75
    # Calculate the x and y points for each angle
    outer_transition = []
    for angle in angles:
        x_angle = x_center - np.sin(angle) * big_radius
        y_angle = y_center - np.cos(angle) * big_radius
        outer_transition.append((x_angle, y_angle))

    outer_transition = np.array(outer_transition)

    # Starting and ending points
    x = [-1.5]
    y = [-1.5]

    # Append the outer transition points
    x += outer_transition[:, 0].tolist()
    y += outer_transition[:, 1].tolist()

    # Parameters for circular motion
    r = 0.3  # Radius of the circular trajectory
    tf_circle = 1  # Total time for circular motion
    omega = -2 * np.pi / tf_circle  # Angular velocity (rad/s)
    xc = 0  # Center x-coordinate
    yc = -3 + r  # Center y-coordinate

    # Generate circular trajectory
    theta1_sequence, theta2_sequence = generate_circular_trajectory(r, omega, xc, yc, tf_circle, num_points=10)
    
    # Append the circular trajectory points
    x += theta1_sequence
    y += theta2_sequence

    # Reverse the outer transition and multiply by [-1, 1] to reflect the back transition
    outer_transition = outer_transition[::-1] * [-1, 1]
    
    # Append the back transition points
    x += outer_transition[:, 0].tolist()
    y += outer_transition[:, 1].tolist()
    x.append(1.5)
    y.append(-1.5)
    

    # Create parameter t (cumulative distance as the parameter)
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)  # Calculate distances between consecutive points
    t = np.concatenate(([0], np.cumsum(distances)))  # Cumulative sum to represent parameter t
    t = t / t[-1]  # Normalize t to be in range [0, 1]

    # Create parametric cubic splines for x(t) and y(t)
    spline_x = CubicSpline(t, x, bc_type='clamped')
    spline_y = CubicSpline(t, y, bc_type='clamped')

    # Generate new parameter values for a smooth curve
    t_new = np.linspace(0, 1, 4000)
    x_spline = spline_x(t_new)
    y_spline = spline_y(t_new)

    # Constrain the points to stay within the circle
    distances_spline = np.sqrt(x_spline**2 + y_spline**2)
    mask = distances_spline > radius
    x_spline[mask] = x_spline[mask] * (radius - 1e-9) / distances_spline[mask] 
    y_spline[mask] = y_spline[mask] * (radius - 1e-9) / distances_spline[mask] 
    
    # Plot the generated spline
    plt.plot(x_spline, y_spline, label='Spline', color='r', linewidth=2)
    plt.plot(x, y, 'o', label='Points', color='b')
    
    # Plot the circle
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    plt.plot(x_circle, y_circle, '--', label='Circle', color='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return x_spline, y_spline


# Call the function to generate and plot the spline
generate_spline()


def generate_trajectory(x_ref, y_ref):
    theta1_sequence = []
    theta2_sequence = []
    theta1_conf1 = []
    theta2_conf1 = []
    theta1_conf2 = []
    theta2_conf2 = []
    last_index = None
    
    for i in range(len(x_ref)):
        theta1_c1, theta2_c1 = inverse_kinematics(x_ref[i], y_ref[i], configuration=1)
        theta1_conf1.append(theta1_c1)
        theta2_conf1.append(theta2_c1)
        
        theta1_c2, theta2_c2 = inverse_kinematics(x_ref[i], y_ref[i], configuration=2)
        theta1_conf2.append(theta1_c2)
        theta2_conf2.append(theta2_c2)
    
    # Iterate in reverse
    for i in range(len(theta1_conf1) - 1, -1, -1):
        if theta1_conf1[i] == 0 and theta2_conf1[i] == 0:
            last_index = i  # Store the index
            print(f"Found at index: {i}")
            break
        
    last_index = 2582
    constant_instants = 3000
    # Append the first configuration until last_index and the second configuration after last_index
    theta1_sequence += [theta1_conf1[0]] * constant_instants
    theta2_sequence += [theta2_conf1[0]] * constant_instants
    theta1_sequence += theta1_conf1[:last_index]
    theta2_sequence += theta2_conf1[:last_index]
    theta1_sequence += theta1_conf2[last_index:]
    theta2_sequence += theta2_conf2[last_index:]
    theta1_sequence += [theta1_conf2[-1]] * constant_instants
    theta2_sequence += [theta2_conf2[-1]] * constant_instants
    
    return theta1_sequence, theta2_sequence

def generate_input_sequence(x_reference, K_eq = 44):
    u_reference = np.zeros((1, len(x_reference)))
    u_reference = K_eq * np.sin(x_reference[2, :])
    
    return u_reference

def apply_smoothing(array, window_size=50):
    """Apply a simple moving average smoothing to the input array, leaving the first and last 10 samples unaffected."""
    
    # Create a copy of the input array to avoid modifying the original one
    smoothed_array = array.copy()

    # Perform smoothing on the central part of the array (ignoring the first and last 10 samples)
    smoothed_array[window_size:-window_size] = np.convolve(array, np.ones(window_size)/window_size, mode='same')[window_size:-window_size]
    
    return smoothed_array

def calculate_dtheta1_dtheta2(x_reference):
    dtheta1 = np.diff(x_reference[2, :])/pm.dt
    dtheta2 = np.diff(x_reference[3, :])/pm.dt
    dtheta1 = np.append(dtheta1, dtheta1[-1])
    dtheta2 = np.append(dtheta2, dtheta2[-1])
    
    return dtheta1, dtheta2


def main():
    # Generate the spline
    x_spline, y_spline = generate_spline()
    theta1_sequence, theta2_sequence = generate_trajectory(x_spline, y_spline)
    
    theta1_sequence = apply_smoothing(theta1_sequence)
    theta2_sequence = apply_smoothing(theta2_sequence)

    x_reference = np.zeros((4, len(theta1_sequence)))
    for i in range(len(theta1_sequence)):
        x_reference[2:, i] = [theta1_sequence[i], theta2_sequence[i]]
        
    u_reference = generate_input_sequence(x_reference)

    dtheta1, dtheta2 = calculate_dtheta1_dtheta2(x_reference)
    
    x_reference[0,:] = dtheta1
    x_reference[1,:] = dtheta2
    
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
    plt.plot(u_reference, label=r'$\tau_1^{ref}$', color='r', linewidth=2)
    plt.legend()

    print(x_reference.shape)

    # qdm.save_mpc_trajectory(x_reference, u_reference)

    anim(x_reference.T)
    
    
if __name__ == "__main__":
    main()