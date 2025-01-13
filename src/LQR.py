import numpy as np
from numpy.linalg import inv 
import matplotlib.pyplot as plt
import parameters as pm
from parameters import state_perturbation_percentage, affine_perturbation
import dynamics as dyn
import costTask3 as cost


def LQR_system_regulator(x_gen, u_gen):
    """
    Applies LQR regulation to a system and plots its evolution.

    Args:
        x_gen (np.ndarray): Generated state trajectory.
        u_gen (np.ndarray): Generated control trajectory.

    Returns:
        tuple: x_evolution_after_LQR (np.ndarray), u_regulator (np.ndarray).
    """
    print('\n\n\
        \t------------------------------------------\n \
        \t\tLaunching: LQR Tracker\n \
        \t------------------------------------------')
    
    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    TT = x_gen.shape[1]

    x_regulator = np.zeros((x_size, TT))
    u_regulator = np.zeros((u_size, TT))
    x_natural_evolution = np.zeros((x_size, TT))
    x_evolution_after_LQR = np.zeros((x_size, TT))

    x_regulator[:, 0]         = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_natural_evolution [:,0] = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_evolution_after_LQR[:,0]= x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation

    u_regulator = u_gen

    delta_x = x_regulator - x_gen
    delta_u = u_regulator - u_gen

    # Initialize the perturberd system as the natural evolution of the system
    # without a proper regulation
    for t in range(TT-1):
        x_natural_evolution[:, t+1] = dyn.dynamics(x_natural_evolution[:, t], u_regulator[:,t])


    Qt = np.zeros((x_size, x_size,TT-1))
    Rt = np.zeros((u_size, u_size,TT-1))

    K_Star = np.zeros((u_size, x_size, TT-1))

    A = np.zeros((x_size,x_size,TT-1))
    B = np.zeros((x_size,u_size,TT-1))

    for t in range(TT-1):
        A[:,:,t] = dyn.jacobian_x_new_wrt_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = dyn.jacobian_x_new_wrt_u(x_gen[:,t])
        Qt[:,:,t] = cost.hessian1_J(t)           
        Rt[:,:,t] = cost.hessian2_J(t)              
    QT = cost.hessian_terminal_cost()

    K_Star = LQR_solver(A, B, Qt, Rt, QT)

    for t in range(TT-1):
        delta_u[:, t]  = K_Star[:,:,t] @ delta_x[:, t]
        delta_x[:, t+1]= A[:,:,t] @ delta_x[:,t] + B[:,:,t] @ delta_u[:, t]
    
    for t in range(TT-1):
        u_regulator[:,t] = u_gen[:,t] + delta_u[:,t]
        x_regulator[:,t] = x_gen[:,t] + delta_x[:,t]
        x_evolution_after_LQR[:,t+1] = dyn.dynamics(x_evolution_after_LQR[:,t], u_regulator[:,t])

    plt.figure()
    for i in range(x_size):
        plt.plot(x_evolution_after_LQR[i, :], color = 'red', label = f'x[{i}]')
    plt.plot(u_regulator[0,:], color = 'blue', label = 'u_regulator')
    plt.title("System Evolution with Real Dynamics and LQRegulated input")
    plt.legend()
    plt.grid()
    plt.show()

    # NOTA: con delta_x e delta_u, l'LQR è perfettamente in grado di annullare la distanza dalla reference
    
    plt.figure()
    for i in range(x_size):
        plt.plot(delta_x[i, :], color = 'red', label = r'$\Delta$' f'x[{i}]')
    plt.plot(delta_u[0,:], color = 'blue', label = r'$\Delta$' 'u')
    plt.title("LQR Residuals evolution")
    plt.legend()
    plt.grid()
    plt.show()

    return x_evolution_after_LQR, delta_u


def LQR_solver(A, B, Qt_Star, Rt_Star, QT_Star):
    """
    Solves the discrete-time Linear Quadratic Regulator (LQR) problem.

    Args:
        A (np.ndarray): State transition matrices over time.
        B (np.ndarray): Control matrices over time.
        Qt_Star (np.ndarray): State cost matrices over time.
        Rt_Star (np.ndarray): Control cost matrices over time.
        QT_Star (np.ndarray): Terminal state cost matrix.

    Returns:
        np.ndarray: Feedback gain matrices (K).
    """
    x_size = A.shape[0]
    u_size = B.shape[1]
    TT = A.shape[2]+1
    
    delta_x = np.zeros((x_size,TT))

    P = np.zeros((x_size,x_size,TT))
    Pt = np.zeros((x_size,x_size))
    Ptt= np.zeros((x_size,x_size))
    
    K = np.zeros((u_size,x_size,TT-1))
    Kt= np.zeros((u_size,x_size))

    ######### Solve the Riccati Equation [S6C4]
    P[:,:,-1] = QT_Star

    for t in reversed(range(TT-1)):
        At  = A[:,:,t]
        Bt  = B[:,:,t]
        Qt  = Qt_Star[:,:,t]
        Rt  = Rt_Star[:,:,t]
        Ptt = P[:,:,t+1]

        temp = (Rt + Bt.T @ Ptt @ Bt)
        inv_temp = inv(temp)
        Kt =-inv_temp @ (Bt.T @ Ptt @ At)
        Pt = At.T @ Ptt @ At + At.T@ Ptt @ Bt @ Kt + Qt

        K[:,:,t] = Kt
        P[:,:,t] = Pt 
    return K
    
    
def plot_trajectories(x_real_LQR, u_real_LQR, x_gen, u_gen):
    """
    Plot the system trajectories comparing real and desired states/inputs.
    Each variable is plotted in a separate subplot with its reference.
    
    Args:
        x_real_mpc: Real state trajectory (4xT array)
        u_real_mpc: Real input trajectory (1xT array)
        x_gen: Desired optimal state trajectory (4xT array)
        u_gen: Desired optimal input trajectory (1xT array)
    """
    # Define naming and color schemes
    names = {
        0: r'$\dot \theta_1$', 1: r'$\dot \theta_2$', 
        2: r'$\theta_1$', 3: r'$\theta_2$', 4: r'$\tau_1$'
    }
    colors_ref = {0: 'm', 1: 'orange', 2: 'b', 3: 'g', 4: 'r'}
    colors_gen = {0: 'darkmagenta', 1: 'chocolate', 2: 'navy', 3: 'limegreen', 4: 'darkred'}
    
    T = x_real_LQR.shape[1]
    k = np.arange(T)
    
    # Create figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(6, 10))
    fig.suptitle('System Trajectories: Real vs Reference', fontsize=16)
    
    # Plot states
    for i in range(4):
        axs[i].plot(k, x_real_LQR[i,:], color=colors_ref[i], linestyle='-', linewidth=2, 
                   label=f'{names[i]}')
        axs[i].plot(k, x_gen[i,:], color=colors_gen[i], linestyle='--', linewidth=2,
                   label=f'{names[i]}' + r'$^{des}$')
        axs[i].set_ylabel('Angular Velocity (rad/s)' if i < 2 else 'Angle (rad)')
        axs[i].legend()
        axs[i].grid(True)
    
    # Plot input
    axs[4].plot(k, u_real_LQR[0,:], color=colors_ref[4], linestyle='-', linewidth=2,
                label=f'{names[4]}')
    axs[4].plot(k, u_gen[0,:], color=colors_gen[4], linestyle='--', linewidth=2,
                label=f'{names[4]}' + r'$^{des}$')
    axs[4].set_ylabel(r'Torque (N$\cdot$m)')
    axs[4].set_xlabel(r'Time $t$')
    axs[4].legend()
    axs[4].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return


def plot_tracking_errors(x_real_LQR, x_gen, delta_u):
    """
    Plot individual state tracking errors and control input over time.
    
    Args:
        x_real_mpc: Real state trajectory (4xT array)
        x_gen: Desired optimal state trajectory (4xT array)
        u_real_mpc: Real control input trajectory (1xT array)
        u_gen: Desired optimal control input trajectory (1xT array)
    """
    # Define naming and color schemes
    names = {
        0: r'$\dot \theta_1$', 1: r'$\dot \theta_2$', 
        2: r'$\theta_1$', 3: r'$\theta_2$',
        4: r'$\tau_1$'
    }
    colors = {0: 'm', 1: 'orange', 2: 'b', 3: 'g', 4: 'r'}
    
    T = x_real_LQR.shape[1]
    time = np.arange(T)
    
    # Create figure with 5 subplots
    fig, axs = plt.subplots(5, 1, figsize=(6, 10))
    fig.suptitle('State Tracking Errors and Control Input', fontsize=16)
    
    # Plot individual state tracking errors
    for i in range(4):
        error = (x_real_LQR[i,:] - x_gen[i,:])
        axs[i].plot(time, error, color=colors[i], linestyle='-', linewidth=2,
                   label=f'{names[i]} - {names[i]}' + r'$^{ref}$')
        
        # Set appropriate labels
        if i < 2:
            axs[i].set_ylabel('Angular Velocity (rad/s)')
        else:
            axs[i].set_ylabel('Angle (rad)')
        axs[i].legend()
        axs[i].grid(True)
    
    # Plot control input
    error = delta_u[0,:]
    axs[4].plot(time,error, color=colors[4], linestyle='-', linewidth=2,
                label=f'{names[4]}')
    axs[4].set_ylabel(r'Torque (N$\cdot$m)')
    axs[4].set_xlabel(r'Time $t$')
    axs[4].legend()
    axs[4].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    return




