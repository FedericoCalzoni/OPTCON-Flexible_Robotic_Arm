from dynamics import dynamics as dyn
from dynamics import jacobian
from visualizer import animate_double_pendulum as anim
import numpy as np
from parameters import L1, L2, t_i, t_f, dt
from newton_method import newton_method
from matplotlib import pyplot as plt

def main():
    """
    Main function to simulate the dynamics and visualize the results.
    """
    # Initial state and input
    x_0 = np.array([[0], [0], [np.pi/6], [np.pi]])  # Initial state (dtheta1, dtheta2, theta1, theta2)
    u_0 = np.array([[10], [0], [0], [0]])  # Input (tau1, tau2 , - ,  - )
    
    # Find equilibria
    
    z_0_eq1 = np.array([[-np.pi/3], [+np.pi/3], [-8]])
    z_0_eq2 = np.array([[np.pi/3], [-np.pi/3], [8]])
    equilibria_1 = newton_method(z_0_eq1, jacobian)
    equilibria_2 = newton_method(z_0_eq2, jacobian)
    eq1_0 = equilibria_1[0].item()
    eq1_1 = equilibria_1[1].item()
    eq1_2 = equilibria_1[2].item()
    eq2_0 = equilibria_2[0].item()
    eq2_1 = equilibria_2[1].item()
    eq2_2 = equilibria_2[2].item()
    print("eq: ", equilibria_2)
    
    x_0 = np.array([[0], [0], [eq1_0], [eq1_1]])
    u_0 = np.array([[eq1_2], [0], [0], [0]])

    # Compute dynamics for each time step
    time_intervals = int((t_f - t_i) / dt + 1)
    x_history = [x_0]
    print("Computing dynamics...")
    for i in range(time_intervals):
        x_history.append(dyn(x_history[i], u_0, dt))
    
    # x_e = np.zeros((time_intervals, 4))
    # for i in range(1, time_intervals):
    #     x_e[i] = (x_history[i]).flatten() -(x_history[i-1]).flatten()
    
    # plt.plot(x_e)
    
    # Convert state history to a matrix
    matrix_x_history = np.hstack(x_history)

    # Visualize the simulation
    anim(matrix_x_history.T, L1, L2, frame_skip=1)

if __name__ == "__main__":
    main()
