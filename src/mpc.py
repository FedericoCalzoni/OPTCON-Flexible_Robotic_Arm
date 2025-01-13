import numpy as np
import cvxpy as cp
from dynamics import dynamics, jacobian_x_new_wrt_x, jacobian_x_new_wrt_u
import matplotlib.pyplot as plt
import costTask4 as cost
from parameters import (
    T_pred, u_max, u_min, x_dtheta_max, x_dtheta_min,
    x_theta1_max, x_theta1_min, x_theta2_max, x_theta2_min,
    state_initial_perturbation, noise_sensor, noise_actuator
)

def solver_linear_mpc(A, B, Q, R, x_t, t_start, x_gen, u_gen): 
    x_size, u_size, T = B.shape
    
    # Pre-compute reference trajectories for the prediction horizon
    t_indices = np.minimum(t_start + np.arange(T_pred), T-1)
    x_ref = x_gen[:, t_indices]
    u_ref = u_gen[:, t_indices]
    
    # Get time-varying matrices for the prediction horizon
    A_pred = A[:, :, t_indices[:-1]]
    B_pred = B[:, :, t_indices[:-1]]
    Q_pred = Q[:, :, t_indices[:-1]]
    R_pred = R[:, :, t_indices[:-1]]
    
    # Define decision variables
    x_mpc = cp.Variable((x_size, T_pred))
    u_mpc = cp.Variable((u_size, T_pred))
    
    # Initialize constraints list with pre-allocated capacity
    constraints = []
    constraints.append(x_mpc[:, 0] == x_t)
    
    # Vectorized state and input constraints for entire horizon
    constraints.extend([
        cp.vstack([u_mpc[:, tau] for tau in range(T_pred-1)]) <= u_max,
        cp.vstack([u_mpc[:, tau] for tau in range(T_pred-1)]) >= u_min,
        cp.vstack([x_mpc[0:2, tau] for tau in range(T_pred-1)]) <= x_dtheta_max,
        cp.vstack([x_mpc[0:2, tau] for tau in range(T_pred-1)]) >= x_dtheta_min,
        cp.vstack([x_mpc[2, tau] for tau in range(T_pred-1)]) <= x_theta1_max,
        cp.vstack([x_mpc[2, tau] for tau in range(T_pred-1)]) >= x_theta1_min,
        cp.vstack([x_mpc[3, tau] for tau in range(T_pred-1)]) <= x_theta2_max,
        cp.vstack([x_mpc[3, tau] for tau in range(T_pred-1)]) >= x_theta2_min
    ])
    
    # Add dynamics constraints using matrix operations
    for tau in range(T_pred - 1):
        constraints.append(
            x_mpc[:, tau + 1] == A_pred[:,:,tau] @ x_mpc[:, tau] + B_pred[:,:,tau] @ u_mpc[:, tau]
        )
    
    # Compute cost using vectorized operations
    cost = 0
    for tau in range(T_pred - 1):
        delta_x = x_mpc[:, tau] - x_ref[:, tau]
        delta_u = u_mpc[:, tau] - u_ref[:, tau]
        cost += cp.quad_form(delta_x, Q_pred[:,:,tau]) + cp.quad_form(delta_u, R_pred[:,:,tau])
    
    # Terminal cost
    delta_x_terminal = x_mpc[:, -1] - x_ref[:, -1]
    cost += cp.quad_form(delta_x_terminal, Q[:, :, -1])

    # Solve optimization problem with warm start if available
    problem = cp.Problem(cp.Minimize(cost), constraints)
    
    # Try OSQP first as it's typically faster for MPC problems
    try:
        problem.solve(solver='OSQP', warm_start=True)
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return x_mpc.value, u_mpc.value, problem
    except cp.error.SolverError:
        pass
        
    # Fall back to ECOS if OSQP fails
    try:
        problem.solve(solver='ECOS')
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return x_mpc.value, u_mpc.value, problem
    except cp.error.SolverError:
        print("Both OSQP and ECOS failed to solve the problem.")
        return None, None, problem

def compute_mpc(x_gen, u_gen):
    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    T = x_gen.shape[1]
    
    # Pre-allocate arrays
    A = np.zeros((x_size, x_size, T))
    B = np.zeros((x_size, u_size, T))
    Q = np.zeros((x_size, x_size, T))
    R = np.zeros((u_size, u_size, T))
    x_real_mpc = np.zeros((x_size, T))
    u_real_mpc = np.zeros((u_size, T))
    
    x_real_mpc[:,0] = x_gen[:,0] * (1 + state_initial_perturbation)
    
    # Pre-compute system matrices and cost matrices
    for t in range(T-1):
        A[:,:,t] = jacobian_x_new_wrt_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = jacobian_x_new_wrt_u(x_gen[:,t])
        Q[:,:,t] = cost.hessian1_J(t)           
        R[:,:,t] = cost.hessian2_J(t)              
    Q[:,:,-1] = cost.hessian_terminal_cost()
    
    # Generate sensor noise matrix for entire trajectory
    sensor_noise = noise_sensor * np.random.randn(x_size, T)
    actuator_noise = noise_actuator * np.random.randn(u_size, T)
    
    for t in range(T-1):
        x_t = x_real_mpc[:,t] + sensor_noise[:,t]
        
        x_mpc_t, u_mpc_t, problem = solver_linear_mpc(A, B, Q, R, x_t, t, x_gen, u_gen)
        
        if x_mpc_t is None:
            print(f"MPC failed at time {t}")
            u_mpc_t = np.zeros((u_size, 1))
            # break
            
        u_real_mpc[:,t] = u_mpc_t[:,0] + actuator_noise[:,t]
        x_real_mpc[:,t+1] = dynamics(x_real_mpc[:,t], u_real_mpc[:,t])
        
        if t % 20 == 0: 
            tracking_error_pos = np.linalg.norm(x_real_mpc[2:4,t] - x_gen[2:4,t])
            tracking_error_vel = np.linalg.norm(x_real_mpc[0:2,t] - x_gen[0:2,t])
            print(f"t={t}")
            print(f"Position error={tracking_error_pos:.4f}")
            print(f"Velocity error={tracking_error_vel:.4f}")
            print(f"Real state:   dtheta1={x_real_mpc[0,t]:.2f}, dtheta2={x_real_mpc[1,t]:.2f}, " 
                  f"theta1={x_real_mpc[2,t]:.2f}, theta2={x_real_mpc[3,t]:.2f}")
            print(f"Sensor state:    dtheta1={x_t[0]:.2f}, dtheta2={x_t[1]:.2f}, "
                  f"theta1={x_t[2]:.2f}, theta2={x_t[3]:.2f}")
            print(f"Reference state: dtheta1={x_gen[0,t]:.2f}, dtheta2={x_gen[1,t]:.2f}, "
                  f"theta1={x_gen[2,t]:.2f}, theta2={x_gen[3,t]:.2f}")
            print(f"Real input:      tau1={u_real_mpc[0,t]:.2f}")
            print(f"MPC  input:      tau1={u_mpc_t[0,0]:.2f}")
            print(f"Reference input: tau1={u_gen[0,t]:.2f}")
            if problem.value is not None:
                print(f"cost={problem.value:.4f}")
            print("---")
    
    return x_real_mpc, u_real_mpc


def plot_trajectories(x_real_mpc, u_real_mpc, x_gen, u_gen):
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
    
    T = x_real_mpc.shape[1]
    k = np.arange(T)
    
    # Create figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(6, 10))
    fig.suptitle('System Trajectories: Real vs Reference', fontsize=16)
    
    # Plot states
    for i in range(4):
        axs[i].plot(k, x_real_mpc[i,:], color=colors_ref[i], linestyle='-', 
                   label=f'{names[i]}')
        axs[i].plot(k, x_gen[i,:], color=colors_gen[i], linestyle='--', 
                   label=f'{names[i]}' + r'$^{des}$')
        axs[i].set_ylabel('Angular Velocity (rad/s)' if i < 2 else 'Angle (rad)')
        axs[i].legend()
        axs[i].grid(True)
    
    # Plot input
    axs[4].plot(k, u_real_mpc[0,:], color=colors_ref[4], linestyle='-', 
                label=f'{names[4]}')
    axs[4].plot(k, u_gen[0,:], color=colors_gen[4], linestyle='--', 
                label=f'{names[4]}' + r'$^{des}$')
    axs[4].set_ylabel(r'Torque (N$\cdot$m)')
    axs[4].set_xlabel(r'Time $t$')
    axs[4].legend()
    axs[4].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return


def plot_tracking_errors(x_real_mpc, x_gen, u_real_mpc, u_gen):
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
    
    T = x_real_mpc.shape[1]
    time = np.arange(T)
    
    # Create figure with 5 subplots
    fig, axs = plt.subplots(5, 1, figsize=(6, 10))
    fig.suptitle('State Tracking Errors and Control Input', fontsize=16)
    
    # Plot individual state tracking errors
    for i in range(4):
        error = (x_real_mpc[i,:] - x_gen[i,:])
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
    error = (u_real_mpc[0,:] - u_gen[0,:])
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