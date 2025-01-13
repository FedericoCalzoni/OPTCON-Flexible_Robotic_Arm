import dynamics as dyn
from visualizer import animate_double_pendulum as anim
import numpy as np
import parameters as pm
from newton_method import newton_method
import reference_trajectory
from newton_opt_ctrl import newton_for_optcon as newton_OC
from newton_opt_ctrl import plot_optimal_trajectory
import matplotlib.pyplot as plt
import data_manager as dm
from mpc import compute_mpc
from LQR import LQR_system_regulator as LQR
from multi_equilibria_trajectory import multi_equilibria_trajectory

def main():
    print("\n\n\
          \t ------------------------------------------\n\
          \t\t\t ♫ START ♪\n\
          \t ------------------------------------------\n\
          \n\n")

    # IMPORTANT: The variable "Task_to_run" make you able to select
    # the tasks to be run. 
    Task_to_run = [2,3, 4]

    #####################################
    ##           Task 1 - 2            ##
    #####################################
    if pm.optimal_trajectory_given:
        x_gen, u_gen = dm.load_optimal_trajectory(version = '30')
        x_reference, u_reference = dm.load_reference_trajectory(version = '50')
        plt.figure()
        delta_x = np.zeros((4,pm.TT))
        for i in range(x_reference.shape[0]):
            delta_x[i, :] =  x_gen[i, :] - x_reference[i, :] 
            plt.plot(delta_x[i, :], label = f'delta x{i}')
        plt.legend()
        plt.grid()
        plt.show()
        plt.figure()
        delta_u = u_gen - u_reference
        plt.plot(delta_u[0, :], label = 'delta u')
        plt.legend()
        plt.grid()
        plt.show()
    elif 1 in Task_to_run:
        # Compute two equilibria and make a step transition between them
        z_0_eq1 = np.array([[-np.pi/2+0.01], [np.pi/2-0.01], [-44]])
        z_0_eq2 = np.array([[np.pi/2-0.01], [-np.pi/2+0.01], [+44]])
        # z_0_eq1 = np.array([[np.pi+np.pi/2.1], [-np.pi/2.1], [-40]])
        # z_0_eq2 = np.array([[np.pi-np.pi/2.1], [+np.pi/2.1], [+40]])
        x_eq0, u_eq0 = newton_method(z_0_eq1)
        x_eqf, u_eqf = newton_method(z_0_eq2)

        # Initial state and input
        print("Initial state:\t", x_eq0.T ,"\tInitial input:\t", u_eq0.T)
        print("Final state:\t", x_eqf.T, "\tFinal Input:\t", u_eqf.T)
        x_reference, u_reference = reference_trajectory.generate_trajectory(x_eq0, x_eqf, u_eq0, u_eqf)
        plt.plot(x_reference[2,:]) 
        plt.show()
        x_gen, u_gen, l = newton_OC(x_reference, u_reference)
        dm.save_optimal_trajectory(x_gen, u_gen)
    elif 2 in Task_to_run:
        x_reference, u_reference = reference_trajectory.generate_smooth_trajectory()
        dm.save_reference_trajectory(x_reference, u_reference)    
        x_gen, u_gen, l = newton_OC(x_reference, u_reference)
        dm.save_optimal_trajectory(x_gen, u_gen)
        delta_x = np.zeros((4,pm.TT))
        for i in range(x_reference.shape[0]):
            delta_x[i, :] =  x_gen[i, :] - x_reference[i, :] 
            plt.plot(delta_x[i, :], label = f'delta x{i}')
        delta_u = u_gen - u_reference
        plt.plot(delta_u[0, :], label = 'delta u')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        raise ("Neither Task 1 or Task 2 has been selected.")
    anim(x_gen.T)


    #####################################
    ##              Task 3             ##
    #####################################
    if pm.LQR_trajectory_given:
        x_LQR, u_LQR = dm.load_lqr_trajectory(version = 'latest')
    else:
        x_LQR, u_LQR = LQR(x_gen, u_gen)
        dm.save_lqr_trajectory(x_LQR, u_LQR)
    
    plt.figure()
    for i in range(4):
        plt.plot(x_LQR[i, :], color = 'red', label = f'x_LQR[{0}]' )
        plt.plot(x_gen[i, :], color = 'blue', label = f'x_gen[{0}]' )
    plt.title('Optimal Trajectory VS LQR Trajectory')
    plt.grid()
    plt.legend()
    plt.show(block = True)

    anim(x_LQR.T)

    #####################################
    ##              Task 4             ##
    #####################################

    if pm.MPC_trajectory_given:
        dm.load_mpc_trajectory(version = 'latest')
    else:
        x_mpc, u_mpc = compute_mpc(x_gen, u_gen)
        dm.save_mpc_trajectory(x_mpc, u_mpc)

    
    anim(x_mpc.T)

if __name__ == "__main__":
    main()
