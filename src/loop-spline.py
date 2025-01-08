import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
import parameters as pm 
from visualizer import animate_double_pendulum as anim


# Inverse Kinematics Function
def inverse_kinematics(x, y, L1=pm.L1, L2=pm.L2):
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


def generate_circular_trajectory(r, omega, xc, yc, tf_circle, num_points=100):
    t = np.linspace(0, tf_circle, num_points)
    theta = omega * t
    x = xc + r * np.sin(theta+np.pi)
    y = yc + r * np.cos(theta+np.pi)
    return x.tolist(), y.tolist()

def generate_spline():
    angle = np.pi / 8
    radius = 3  # Circle radius
    help_radius = 2.8  

    # Points for the initial and final sections
    x_angle = -np.sin(angle) * help_radius
    y_angle = -np.cos(angle) * help_radius

    x = [-1.5]
    y = [-1.5]

    x.append(x_angle)
    y.append(y_angle)

    # Parameters for circular motion
    r = 0.25  # Radius of the circular trajectory
    tf_circle = 1  # Total time for circular motion
    omega = -2 * np.pi / tf_circle  # Angular velocity (rad/s)
    xc = 0  # Center x-coordinate
    yc = -2.9 + r  # Center y-coordinate

    # Generate circular trajectory
    theta1_sequence, theta2_sequence = generate_circular_trajectory(r, omega, xc, yc, tf_circle, num_points=10)
    
    x += theta1_sequence
    y += theta2_sequence

    x.append(-x_angle)
    y.append(y_angle)

    x.append(1.5)
    y.append(-1.5)

    # Parameter t (cumulative distance as the parameter)
    t = np.linspace(0, 1, len(x))

    # Create parametric cubic splines for x(t) and y(t)
    spline_x = CubicSpline(t, x, bc_type='clamped')
    spline_y = CubicSpline(t, y, bc_type='clamped')
    # spline_x = interp1d(t, x, kind=1)
    # spline_y = interp1d(t, y, kind=1)

    # Generate new parameter values for a smooth curve
    t_new = np.linspace(0, 1, 500)
    x_spline = spline_x(t_new)
    y_spline = spline_y(t_new)

    # Constrain the points to stay within the circle
    distances = np.sqrt(x_spline**2 + y_spline**2)
    mask = distances > radius
    x_spline[mask] = x_spline[mask] * radius / distances[mask]
    y_spline[mask] = y_spline[mask] * radius / distances[mask]

    # Plot the parametric spline
    alpha = np.linspace(0, 2 * np.pi, 500)  # Circle for visualization
    circle_x = radius * np.cos(alpha)
    circle_y = radius * np.sin(alpha)

    plt.figure(figsize=(8, 6))
    plt.plot(circle_x, circle_y, '--', label='Circle Boundary', color='green')  # Circle boundary
    plt.plot(x, y, 'o', label='Data Points', color='red')  # Original points
    plt.plot(x_spline, y_spline, label='Constrained Spline', color='blue')  # Spline curve
    plt.title('Constrained Parametric Cubic Spline')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensures correct aspect ratio
    plt.show()

    return x_spline, y_spline

# Call the function to generate and plot the spline
generate_spline()


def generate_trajectory(x_ref, y_ref):
    theta1_sequence = []
    theta2_sequence = []
    for i in range(len(x_ref)):
        theta1, theta2 = inverse_kinematics(x_ref[i], y_ref[i])
        theta1_sequence.append(theta1)
        theta2_sequence.append(theta2)
        
    return theta1_sequence, theta2_sequence

# Generate the spline
x_spline, y_spline = generate_spline()
theta1_sequence, theta2_sequence = generate_trajectory(x_spline, y_spline)

x_reference = np.zeros((4, len(theta1_sequence)))

for i in range(len(theta1_sequence)):
    x_reference[2, i] = theta1_sequence[i]
    x_reference[3, i] = theta2_sequence[i]

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
# plt.plot(u_reference[0, :], label=r'$\tau_1^{ref}$', color='r', linewidth=2)
plt.legend()

anim(x_reference.T)


