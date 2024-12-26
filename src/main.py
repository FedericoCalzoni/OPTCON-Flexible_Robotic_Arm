import dynamics as dyn
from dynamics import jacobian
from visualizer import animate_double_pendulum as anim
import numpy as np
import parameters as param #TODO: import individually L1, L2, t_i, t_f, dt, A0, B0
from newton_method import newton_method
import cost
from gradient_method import gradient_method
import reference_trajectory
import LQR

def main():
    
    z_0_eq1 = np.array([[-np.pi/3], [+np.pi/3], [-8]])
    z_0_eq2 = np.array([[np.pi/3], [-np.pi/3], [8]])
    equilibria_1 = newton_method(z_0_eq1, jacobian)
    equilibria_2 = newton_method(z_0_eq2, jacobian)
    eq1_theta1 = equilibria_1[0].item()
    eq1_theta2 = equilibria_1[1].item()
    eq1_tau1 = equilibria_1[2].item()
    eq2_theta1 = equilibria_2[0].item()
    eq2_theta2 = equilibria_2[1].item()
    eq2_tau1 = equilibria_2[2].item()
    print("eq_1: ", equilibria_1)
    print("eq_2: ", equilibria_2)
    
    # Initial state and input
    x_0 = np.array([[0], [0], [eq1_theta1], [eq1_theta2]]) # Initial state (dtheta1, dtheta2, theta1, theta2)
    u_0 = np.array([[eq1_tau1], [0], [0], [0]]) # Input (tau1, tau2 , - ,  - )
    
    x_size = x_0.size
    u_size = u_0.size
    
    # Final desired state and input
    x_f = np.array([[0], [0], [eq2_theta1], [eq2_theta2]])
    # u_f = np.array([[eq2_tau1], [0], [0], [0]])
    
    print("Initial state: ", x_0)
    print("Initial input: ", u_0)

    # # Compute dynamics for each time step
    # time_intervals = int((t_f - t_i) / dt + 1)
    # x_history = [x_0]
    # print("Computing dynamics...")
    # for i in range(time_intervals):
    #     x_history.append(dyn.dynamics(x_history[i], u_0, dt)[0])
    
    k_eq = 2.0
    smooth_period = 0
    x_reference, u_reference = reference_trajectory.generate_trajectory(param.t_f, x_0, x_f, k_eq, smooth_period, param.dt)
    # reference_trajectory.plot_trajectory(x_reference, u_reference, dt=param.dt)
    
    
    T = x_reference.shape[1]
    
    # # Compute optimal trajectory
    # # Initial guess
    # x_init = np.zeros((x_size, T))
    # u_init = np.zeros((u_size, T))
    # x_size = x_0.size
    # u_size = u_0.size
    # x_optimal, u_optimal, J, lmbd = gradient_method(x_init, u_init, x_reference, u_reference)
    
    
    # initial trajectories guesses
    # x0_trajectory = np.repeat(x_0, T+1, axis=1)
    # u0_trajectory = np.repeat(u_0, T, axis=1)    
    
    x0_trajectory = x_reference
    u0_trajectory = u_reference
    x_trajectory, u_trajectory = LQR.compute_LQR_trajectory(param.A0, param.B0, x0_trajectory, u0_trajectory, x_reference, u_reference, step_size=0.1, max_iter=2)

    # Visualize the simulation
    # matrix_x_history = np.hstack(x_trajectory)
    anim(x_trajectory.T, param.L1, param.L2, frame_skip=1)

if __name__ == "__main__":
    main()
