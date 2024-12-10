from dynamics import dynamics as dyn
from visualizer import animate_double_pendulum as anim
import numpy as np

# Pendulum lengths
L1 = 1.5
L2 = 1.5

# Simulation parameters
t_i = 0
t_f = 10
dt = 1e-4

def main():
    """
    Main function to simulate the dynamics and visualize the results.
    """
    # Initial state and input
    x_0 = np.array([[0], [0], [np.pi/6], [np.pi]])  # Initial state (dtheta1, dtheta2, theta1, theta2)
    u_0 = np.array([[10], [0], [0], [0]])  # Input (tau1, tau2 , - ,  - )

    time_intervals = int((t_f - t_i) / dt + 1)

    x_history = [x_0]

    # Compute dynamics for each time step
    print("Computing dynamics...")
    for i in range(time_intervals):
        x_history.append(dyn(x_history[i], u_0, dt))

    # Convert state history to a matrix
    matrix_x_history = np.hstack(x_history)

    # Visualize the simulation
    anim(matrix_x_history.T, L1, L2)

if __name__ == "__main__":
    main()
