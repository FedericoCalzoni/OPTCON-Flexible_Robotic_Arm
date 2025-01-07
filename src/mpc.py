import numpy as np
import cvxpy as cp
from dynamics import dynamics, jacobian_x_new_wrt_x, jacobian_x_new_wrt_u
import cost
from parameters import T_pred, u_max, u_min, x_dtheta_max, x_dtheta_min, x_theta1_max, x_theta1_min, x_theta2_max, x_theta2_min, state_perturbation_percentage

def solver_linear_mpc(A, B, Q, R, x_t, t_start, x_gen, u_gen): 
    x_size, u_size, T = B.shape
    
    # Define decision variables
    x_mpc = cp.Variable((x_size, T_pred))
    u_mpc = cp.Variable((u_size, T_pred))
    delta_x = cp.Variable((x_size, T_pred))
    delta_u = cp.Variable((u_size, T_pred))
    
    # Initialize cost and constraints
    cost = 0
    constraints = []
    
    # Initial state constraint
    constraints += [x_mpc[:, 0] == x_t]

    for tau in range(T_pred - 1):
        # Get time-varying matrices
        t_current = min(t_start + tau, T-1)
        A_tau = A[:,:,t_current]
        B_tau = B[:,:,t_current]
        Q_tau = Q[:,:,t_current]
        R_tau = R[:,:,t_current]
        
        # State tracking cost
        if t_start + tau >= T:
            x_ref = x_gen[:,-1]
            u_ref = u_gen[:,-1]
        else:
            x_ref = x_gen[:,t_start+tau]
            u_ref = u_gen[:,t_start+tau]
        
        constraints += [
            # System dynamics
            delta_x[:, tau] == x_mpc[:, tau] - x_ref,
            delta_u[:, tau] == u_mpc[:, tau] - u_ref,
            delta_x[:, tau +1] == A_tau @ delta_x[:, tau] + B_tau @ delta_u[:, tau],
            # x_mpc[:, tau + 1] == A_tau @ (x_mpc[:, tau]- x_ref) + B_tau @ (u_mpc[:, tau]- u_ref),
            
            # # Input constraints
            # u_mpc[:, tau] <= u_max,
            # u_mpc[:, tau] >= u_min,
            
            # # State constraints
            # # Angular velocity constraints
            # x_mpc[0, tau] <= x_dtheta_max,
            # x_mpc[0, tau] >= x_dtheta_min,
            # x_mpc[1, tau] <= x_dtheta_max,
            # x_mpc[1, tau] >= x_dtheta_min,
            
            # # Angle constraints
            # x_mpc[2, tau] <= x_theta1_max,
            # x_mpc[2, tau] >= x_theta1_min,
            # x_mpc[3, tau] <= x_theta2_max,
            # x_mpc[3, tau] >= x_theta2_min
        ]

        # Add quadratic costs with weighted Q
        cost += cp.quad_form(delta_x[:,tau], Q_tau) + cp.quad_form(delta_u[:,tau], R_tau)
        
    # Terminal cost
    if t_start + T_pred - 1 >= T:
        delta_xT = x_mpc[:, -1] - x_gen[:,-1]
    else:
        delta_xT = x_mpc[:, -1] - x_gen[:,t_start+T_pred-1]
        
    cost += cp.quad_form(delta_xT, Q[:,:,-1])

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    
    try:
        problem.solve()
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem not solved optimally! Status: {problem.status}")
            return None, None, problem
            
    except cp.error.SolverError as e:
        print(f"Solver error: {e}")
        return None, None, problem

    return x_mpc.value, u_mpc.value, problem

def compute_mpc(x_gen, u_gen):
    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    T = x_gen.shape[1]
    
    # Initialize system matrices
    A = np.zeros((x_size, x_size, T))
    B = np.zeros((x_size, u_size, T))
    
    # Precompute all linearizations
    for t in range(T-1):
        A[:,:,t] = jacobian_x_new_wrt_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = jacobian_x_new_wrt_u(x_gen[:,t])
    
    # Use last linearization for final timestep
    # TODO: check if this is correct
    A[:,:,-1] = A[:,:,-2]
    B[:,:,-1] = B[:,:,-2]
    
    # Initialize trajectories
    x_real_mpc = np.zeros((x_size, T))
    u_real_mpc = np.zeros((u_size, T))
    
    # Set initial state with perturbation
    x_real_mpc[:,0] = x_gen[:,0]*(1 + state_perturbation_percentage)
    
    # Get cost matrices
    Q = np.zeros((x_size, x_size, T))
    R = np.zeros((u_size, u_size, T))
    for t in range(T-1):
        Q[:,:,t] = cost.hessian1_J(t)           
        R[:,:,t] = cost.hessian2_J(t)              
    Q[:,:,-1] = cost.hessian_terminal_cost()
    
    # MPC loop
    for t in range(T-1):
        x_t_mpc = x_real_mpc[:,t]
        
        # Solve MPC problem
        x_mpc_t, u_mpc_t, problem = solver_linear_mpc(A, B, Q, R, x_t_mpc, t, x_gen, u_gen)
        
        if x_mpc_t is None:
            print(f"MPC failed at time {t}")
            break
            
        u_real_mpc[:,t] = u_mpc_t[:,0]
        x_real_mpc[:,t+1] = dynamics(x_real_mpc[:,t], u_real_mpc[:,t])
        
        # Print progress
        if t % 10 == 0: 
            tracking_error_pos = np.linalg.norm(x_real_mpc[2:4,t] - x_gen[2:4,t])
            tracking_error_vel = np.linalg.norm(x_real_mpc[0:2,t] - x_gen[0:2,t])
            print(f"t={t}")
            print(f"Position error={tracking_error_pos:.4f}")
            print(f"Velocity error={tracking_error_vel:.4f}")
            print(f"Current state: theta1={x_real_mpc[2,t]:.2f}, theta2={x_real_mpc[3,t]:.2f}")
            print(f"Reference: theta1={x_gen[2,t]:.2f}, theta2={x_gen[3,t]:.2f}")
            if problem.value is not None:
                print(f"cost={problem.value:.4f}")
            print("---")
    
    return x_real_mpc, u_real_mpc