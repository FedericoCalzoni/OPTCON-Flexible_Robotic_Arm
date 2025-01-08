import dynamics as dyn
from visualizer import animate_double_pendulum as anim
import numpy as np
import parameters as pm
from newton_method import newton_method
import reference_trajectory
from newton_opt_ctrl import newton_for_optcon as newton_OC
import matplotlib.pyplot as plt
import data_manager as dm
from mpc import compute_mpc
from Task_3 import LQR_system_regulator as LQR

def main():
    print("\n\n\
          \t ------------------------------------------\n\
          \t\t\t ♫ START ♪\n\
          \t ------------------------------------------\n\
          \n\n")

    #####################################
    ##           Task 1 - 2            ##
    #####################################
    if pm.optimal_trajectory_given:
        x_gen, u_gen = dm.load_optimal_trajectory(version = '2')
    else:
        z_0_eq1 = np.array([[-np.pi/2+0.01], [np.pi/2-0.01], [-44]])
        z_0_eq2 = np.array([[np.pi/2-0.01], [-np.pi/2+0.01], [+44]])
        x_eq0, u_eq0 = newton_method(z_0_eq1)
        x_eqf, u_eqf = newton_method(z_0_eq2)
        # Initial state and input
        print("Initial state:\t", x_eq0.T ,"\tInitial input:\t", u_eq0.T)
        print("Final state:\t", x_eqf.T, "\tFinal Input:\t", u_eqf.T)
        x_reference, u_reference = reference_trajectory.generate_trajectory(x_eq0, x_eqf, u_eq0, u_eqf)    

        x_gen, u_gen, l = newton_OC(x_reference, u_reference)
        dm.save_optimal_trajectory(x_gen, u_gen)
    
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


    
    anim(x_mpc.T)

if __name__ == "__main__":
    main()
