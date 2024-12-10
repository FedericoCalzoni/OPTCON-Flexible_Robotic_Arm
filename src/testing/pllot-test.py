import numpy as np

import matplotlib.pyplot as plt

def plot_double_pendulum(theta1, theta2, l1=1.5, l2=1.5):
    # Calculate the positions of the pendulum arms
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # Plot the double pendulum
    plt.figure()
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

# Example usage
theta1 = np.pi / 4  # 45 degrees
theta2 = np.pi / 3  # 60 degrees
plot_double_pendulum(theta1, theta2)