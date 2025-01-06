import numpy as np
import dynamics as dyn
import cost
import parameters as params
import control

np.set_printoptions(linewidth=100)

def solve_ltv_LQR(x0, A_trajectory, B_trajectory, Q_trajectory, R_trajectory, S_trajectory, q_trajectory = None, r_trajectory = None):
    
    T = A_trajectory.shape[2]
    x_size = A_trajectory.shape[0]
    u_size = B_trajectory.shape[1]
    augmented = q_trajectory is not None or r_trajectory is not None
    
    
    # Initialize variables
    P = np.zeros((x_size, x_size, T))
    K_star = np.zeros((u_size, x_size, T))
    sigma_star = np.zeros((u_size, T)) 
    delta_x_star = np.zeros((x_size, T))
    delta_u_star = np.zeros((u_size, T))
       
    # Final condition for P
    P[:,:,-1] = Q_trajectory[:,:,-1]
    if augmented:
        p = np.zeros((x_size, T))
        p[:,-1] = q_trajectory[:,-1]
    
    
    # Backward pass: Solve Riccati equation
    print("Reverse iteration")
    for t in reversed(range(T-1)):
        At, Bt = A_trajectory[:, :, t], B_trajectory[:, :, t]
        Qt, Rt, St = Q_trajectory[:, :, t], R_trajectory[:, :, t], S_trajectory[:, :, t]
        Pt_plus_1 = P[:, :, t+1]

        # Feedback gain
        Mt_inv = np.linalg.inv(Rt + Bt.T @ Pt_plus_1 @ Bt)
        K_star[:, :, t] = - Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St)
        
        if augmented:
            qt, rt, pt_plus_1 = q_trajectory[:, t], r_trajectory[:, t], p[:, t+1]
            sigma_star[:, t] = -Mt_inv @ (rt + Bt.T @ pt_plus_1)
            p[:, t] = At.T @ pt_plus_1 - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ (rt + Bt.T @ pt_plus_1) + qt
        
        if t % 100 == 0:
            print(t)
                
        # P[:, :, t] = At.T @ Pt_plus_1 @ At - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St) + Qt
        P[:,:,t] = Qt + At.T @ Pt_plus_1 @ At - K_star[:,:,t].T @ (Rt + Bt.T @ Pt_plus_1 @ Bt) @ K_star[:,:,t]
        # P[:,:,t] = Qt + At.T @ Pt_plus_1 @ At - (At.T @ Pt_plus_1 @ Bt) @ K_star[:,:,t]
        print(f"t: {t}, P: {P[:,:,t]}")
        
    # Forward pass
    print("Forward iteration")   
    delta_x_star[:, 0] = x0   
    for t in range(T-1):
        delta_u_star[:, t] = K_star[:,:,t] @ delta_x_star[:, t] + sigma_star[:,t]
        delta_x_star[:,t+1] = At @ delta_x_star[:,t] + Bt @ delta_u_star[:, t]

    return K_star, sigma_star, delta_x_star


def compute_LQR_trajectory(x_trajectory, u_trajectory, step_size = 0.1, max_iter = 100):
    
    x_size = x_trajectory.shape[0]
    u_size = u_trajectory.shape[0]
    T = x_trajectory.shape[1]

    A_trajectory = np.zeros((x_size, x_size, T))
    B_trajectory = np.zeros((x_size, u_size, T))
    Q_trajectory = np.zeros((x_size, x_size, T))
    R_trajectory = np.zeros((u_size, u_size, T))
    S_trajectory = np.zeros((u_size, x_size, T))
    q_trajectory = np.zeros((x_size, T))
    r_trajectory = np.zeros((u_size, T))
    
    x0 = x_trajectory[:,0]
    
    x_opt_trajectory = np.repeat(x_trajectory[:, :, np.newaxis] , max_iter+1, axis=2)
    u_opt_trajectory = np.repeat(u_trajectory[:, :, np.newaxis] , max_iter+1, axis=2)
    
    x_opt_trajectory = np.zeros((x_size, T, max_iter+1))
    u_opt_trajectory = np.zeros((u_size, T, max_iter+1))
    
    
    
    for k in range(max_iter):      
        # Linearize the system dynamics and cost function around the current trajectory    
        for t in range(T-1):   
            
            # Linearization of the dynamics
            A_trajectory[:,:,t] = dyn.jacobian_x_dot_wrt_x(x_trajectory[:,t], u_trajectory[:,t])
            B_trajectory[:,:,t] = dyn.jacobian_x_dot_wrt_u(x_trajectory[:,t])
            
            # Cost function derivatives
            Q_trajectory[:,:,t] = cost.hessian1_J()
            R_trajectory[:,:,t] = cost.hessian2_J()
            q_trajectory[:,t] = cost.grad1_J(x_opt_trajectory[:,t,k], x_trajectory[:,t])
            r_trajectory[:,t] = cost.grad2_J(u_opt_trajectory[:,t,k], u_trajectory[:,t])
            
        check_system_stability(A_trajectory)
        # Terminal cost
        Q_trajectory[:,:,-1] = cost.hessian_terminal_cost()
        q_trajectory[:,-1] = cost.grad_terminal_cost(x_opt_trajectory[:,-1,k], x_trajectory[:,-1])
        
        # Solve the LTV LQR problem for the current linearization
        K, sigma, delta_x = solve_ltv_LQR(x0,
                                          A_trajectory,
                                          B_trajectory, 
                                          Q_trajectory, 
                                          R_trajectory, 
                                          S_trajectory, 
                                          q_trajectory, 
                                          r_trajectory)
        
        # Update trajectories
        for t in range(T-1):    
            u_opt_trajectory[:,t,k+1] = u_opt_trajectory[:,t,k] + step_size * (sigma[:,t] + K[:,:,t] @ delta_x[:,t]) \
                + K[:,:,t] @ (x_opt_trajectory[:, t, k+1]-x_opt_trajectory[:,t,k]- step_size * delta_x[:,t]) 
                
            x_opt_trajectory[:,t+1,k+1] = dyn.dynamics(x_opt_trajectory[:,t,k+1], 
                                                       u_opt_trajectory[:,t,k+1],
                                                       dt=params.dt)[0]
            
            # print("K[:,:,t]: ", K[:,:,t])
            # print("sigma[:,t]: ", sigma[:,t])
            # print("delta_x[:,t]: ", delta_x[:,t])
            
            if t % 100 == 0:
                print("delta_x: ", delta_x[:,t])
                print(f"u_opt_trajectory:{u_opt_trajectory[:,t,k+1]} \nx_opt_trajectory:{x_opt_trajectory[:,t+1,k+1]}")
        
        # Check convergence
        if np.linalg.norm(u_opt_trajectory[:, :, k + 1] - u_opt_trajectory[:, :, k]) < 1e-3:
            print("LQR Converged")
            break
        
    return x_opt_trajectory[:,:,k+1], u_opt_trajectory[:,:,k+1]  


def check_system_stability(A_trajectory):
    T = A_trajectory.shape[2]  # Number of time steps
    stable = True  # Assume system is stable unless proven otherwise
    
    for t in range(T):
        A_t = A_trajectory[:, :, t]  # Get A_t at time step t
        
        # Compute the eigenvalues of A_t
        eigenvalues = np.linalg.eigvals(A_t)
        
        # Check if all eigenvalues have magnitude less than 1 (for stability)
        if np.any(np.abs(eigenvalues) >= 1):
            print(f"Instability detected at time step {t} with eigenvalues: {eigenvalues}")
            stable = False  # If any eigenvalue is unstable, set stable to False
    
    if not stable:
        print("The system is not stable at some time steps.")