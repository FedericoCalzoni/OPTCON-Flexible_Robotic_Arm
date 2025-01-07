import dynamics as dyn
from dynamics import  jacobian_gravity
from visualizer import animate_double_pendulum as anim
import numpy as np
import parameters as param
from newton_method import newton_method
from gradient_method import gradient_method
import reference_trajectory
from newton_opt_ctrl import newton_for_optcon as newton_OC
import matplotlib.pyplot as plt
import data_manager as dm


from mpc import compute_mpc
import pickle

def main():
    print("\n\n\
          \t ------------------------------------------\n\
          \t\t\t ♫ START ♪\n\
          \t ------------------------------------------\n\
          \n\n")

    z_0_eq1 = np.array([[-np.pi/2+0.01], [np.pi/2-0.01], [-44]])
    z_0_eq2 = np.array([[np.pi/2-0.01], [-np.pi/2+0.01], [+44]])
    equilibria_1 = newton_method(z_0_eq1)
    equilibria_2 = newton_method(z_0_eq2)
    eq1_theta1 = equilibria_1[0].item()
    eq1_theta2 = equilibria_1[1].item()
    eq1_tau1 = equilibria_1[2].item()
    eq2_theta1 = equilibria_2[0].item()
    eq2_theta2 = equilibria_2[1].item()
    eq2_tau1 = equilibria_2[2].item()
    print("eq_1: ", equilibria_1.T, "\n")
    print("eq_2: ", equilibria_2.T, "\n")

    # Initial state and input
    x_0 = np.array([[0], [0], [eq1_theta1], [eq1_theta2]]) # Initial state (dtheta1, dtheta2, theta1, theta2)
    u_0 = np.array([eq1_tau1]) # Input (tau1)
    #print("Initial state: ", x_0)
    #print("Initial input: ", u_0)
    x_size = x_0.size
    u_size = u_0.size
    
    # Final desired state
    x_f = np.array([[0], [0], [eq2_theta1], [eq2_theta2]])
    #print("Final desired state: ", x_f)

    x_reference, u_reference = reference_trajectory.generate_trajectory(param.t_f, x_0.flatten(), x_f.flatten(), eq1_tau1, eq2_tau1)    
    T = x_reference.shape[1]

    x_gen, u_gen, l = newton_OC(x_reference, u_reference)
    dm.save_optimal_trajectory(x_gen, u_gen)
    anim(x_gen.T)

    
#
    # # Compute optimal trajectory using gradient method
    # # Initial guess
    # x_init = np.zeros((x_size, T))
    # u_init = np.zeros((u_size, T))
    # x_size = x_0.size
    # u_size = u_0.size
    # x_optimal, u_optimal, J, lmbd = gradient_method(x_init, u_init, x_reference, u_reference)
    
    # x_trajectory, u_trajectory = LQR.compute_LQR_trajectory(x_reference, u_reference, step_size=0.1, max_iter=10)
    x_gen, u_gen, l = newton_OC(x_reference, u_reference)
    # Visualize the simulation
    # matrix_x_history = np.hstack(x_trajectory)

    #x_debug = np.zeros((x_size,time_intervals-1))
    #u_debug = np.zeros((u_size,time_intervals-2))
    #A_debug = np.zeros((x_size, x_size, time_intervals-1))
    #B_debug = np.zeros((x_size, u_size, time_intervals-1))
    #x_debug[:, 0] = x_0.flatten()
    #
    #for t in range (time_intervals-2):
    #    u_debug[:, t] = u_0.flatten()
    #    A_debug[:,:,t] = dyn.jacobian_x_new_wrt_x(x_debug[:,t], u_debug[:,t])
    #    B_debug[:,:,t] = dyn.jacobian_x_new_wrt_u(x_debug[:,t])
    #    x_debug[:, t+1] = A_debug[:,:,t] @ x_debug[:,t] + B_debug[:,:,t] @ u_debug[:,t]
    #
    #
    
    # Save both arrays to a file
    with open('arrays.pkl', 'wb') as file:
        pickle.dump((x_gen, u_gen), file)

    print("Arrays saved.")

    # Load the arrays back from the file
    with open('Optimal_Trajectories.pkl', 'rb') as file:
        x_gen, u_gen = pickle.load(file)

    # MPC
    x_trajectory, u_trajectory = compute_mpc(x_gen, u_gen)


    #x_trajectory = x_debug
    anim(x_trajectory.T)

if __name__ == "__main__":
    main()
