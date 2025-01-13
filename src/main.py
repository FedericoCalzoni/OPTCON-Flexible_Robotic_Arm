import dynamics as dyn
from visualizer import animate_double_pendulum as anim
import numpy as np
import parameters as pm
from newton_method import newton_method
import reference_trajectory
from newton_opt_ctrl import newton_for_optcon as newton_OC
from newton_opt_ctrl import plot_optimal_trajectory, plot_norm_grad_J, plot_norm_delta_u, plot_cost_evolution
import matplotlib.pyplot as plt
import data_manager as dm
from mpc import compute_mpc
import mpc
from LQR import LQR_system_regulator as LQR
from LQR import plot_LQR_error

def main():
    print("\n\n\
          \t ------------------------------------------\n\
          \t\t\t ♫ START ♪\n\
          \t ------------------------------------------\n\
          \n\n")

    # IMPORTANT: The variable "Task_to_run" make you able to select
    # the tasks to be run. 
    task_to_run = [4]

    #####################################
    ##           Task 1                ##
    #####################################
    if 1 in task_to_run:
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
        x_reference, u_reference = reference_trajectory.generate_trajectory(x_eq0, x_eqf, u_eq0, u_eqf, smooth_percentage=0, t_f=10)
        reference_trajectory.plot_trajectory(x_reference, u_reference, t_f=10)
        x_gen, u_gen, GradJ_u_history, delta_u_history, l = newton_OC(x_reference, u_reference, guess="first equilibria", task=1)
        plot_norm_grad_J(GradJ_u_history)
        plot_norm_delta_u(delta_u_history)
        plot_cost_evolution(l)
        anim(x_gen.T, title = 'Optimal Trajectory Task 1', speed=2)
       
    #####################################
    ##           Task 2                ##
    #####################################
     
    if 2 in task_to_run:
        # Generate a smooth reference trajectory
        x_reference, u_reference = reference_trajectory.generate_smooth_trajectory()
        dm.save_reference_trajectory(x_reference, u_reference)   
        reference_trajectory.plot_trajectory(x_reference, u_reference)
        
        # Compute the optimal trajectory
        x_gen, u_gen, GradJ_u_history, delta_u_history, l = newton_OC(x_reference, u_reference, guess="reference", task=2)
        dm.save_optimal_trajectory(x_gen, u_gen)
        plot_norm_grad_J(GradJ_u_history)
        plot_norm_delta_u(delta_u_history)
        plot_cost_evolution(l)
        anim(x_gen.T, title = 'Optimal Trajectory Task 2', speed=int(pm.t_f/10))
        plot_optimal_trajectory(x_reference, u_reference, x_gen, u_gen)
    elif pm.optimal_trajectory_given:
        x_gen, u_gen = dm.load_optimal_trajectory(version = 'latest')
        x_reference, u_reference = dm.load_reference_trajectory(version = 'latest')
        anim(x_gen.T, title = 'Optimal Trajectory Task 2', speed=int(pm.t_f/10))
        plot_optimal_trajectory(x_reference, u_reference, x_gen, u_gen)
    else:
        raise ValueError("No trajectory given. Please run task 2 to provide a trajectory.")


    #####################################
    ##              Task 3             ##
    #####################################
    if 3 in task_to_run:
        if pm.LQR_trajectory_given:
            x_LQR, u_LQR = dm.load_lqr_trajectory(version = 'latest')
        else:
            if x_gen.shape[1] != pm.TT:
                raise ValueError("The optimal trajectory is not of the right length.")
            x_LQR, u_LQR = LQR(x_gen, u_gen)
            dm.save_lqr_trajectory(x_LQR, u_LQR)
            plot_LQR_error(x_LQR, x_gen)
            
        anim(x_LQR.T, title='LQR Trajectory', speed=int(pm.t_f/10))

    #####################################
    ##              Task 4             ##
    #####################################

    if 4 in task_to_run:
        if pm.MPC_trajectory_given:
            x_mpc, u_mpc = dm.load_mpc_trajectory(version = 'latest')
        else:
            if x_gen.shape[1] != pm.TT:
                raise ValueError("The optimal trajectory is not of the right length.")
            x_mpc, u_mpc = compute_mpc(x_gen, u_gen)
            dm.save_mpc_trajectory(x_mpc, u_mpc)        
    
        mpc.plot_trajectories(x_mpc, u_mpc, x_gen, u_gen)
        mpc.plot_tracking_errors(x_mpc, x_gen, u_mpc, u_gen)
        anim(x_mpc.T, title='MPC Trajectory', speed=int(pm.t_f/10))

if __name__ == "__main__":
    main()