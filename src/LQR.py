import numpy as np
import dynamics as dyn
import cost
import parameters as params
import matplotlib.pyplot as plt

def solve_ltv_LQR(x0, A_trajectory, B_trajectory, Q_trajectory, R_trajectory, S_trajectory, q_trajectory = None, r_trajectory = None):
    
    augmented = False
    cache = None
    cache_inv = None

    QT = Q_trajectory[:,:,-1]
    T = A_trajectory.shape[2]
    
    try:
        # check if matrix is (.. x .. x T) - 3 dimensional array 
        x_size, lA = A_trajectory.shape[1:]
    except ValueError:
        # if not 3 dimensional array, make it (.. x .. x 1)
        A_trajectory = A_trajectory[:,:,None]
        x_size, lA = A_trajectory.shape[1:]

    try:  
        u_size, lB = B_trajectory.shape[1:]
    except ValueError:
        B_trajectory = B_trajectory[:,:,None]
        u_size, lB = B_trajectory.shape[1:]

    try:
        nQ, lQ = Q_trajectory.shape[1:]
    except ValueError:
        Q_trajectory = Q_trajectory[:,:,None]
        nQ, lQ = Q_trajectory.shape[1:]

    try:
        nR, lR = R_trajectory.shape[1:]
    except ValueError:
        R_trajectory = R_trajectory[:,:,None]
        nR, lR = R_trajectory.shape[1:]

    try:
        nSi, nSs, lS = S_trajectory.shape
    except ValueError:
        S_trajectory = S_trajectory[:,:,None]
        nSi, nSs, lS = S_trajectory.shape

    # Check dimensions consistency -- safety
    if nQ != x_size:
        print("Matrix Q does not match number of states")
        exit()
    if nR != u_size:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != x_size:
        print("Matrix S does not match number of states")
        exit()
    if nSi != u_size:
        print("Matrix S does not match number of inputs")
        exit()


    # Check for affine terms

    if q_trajectory is not None or r_trajectory is not None:
        augmented = True
        qT = q_trajectory[:,-1]

    P = np.zeros((x_size, x_size, T))
    p = np.zeros((x_size, T))
    K_star = np.zeros((u_size, x_size, T))
    sigma_star = np.zeros((u_size, T))
    delta_u_star = np.zeros((u_size, T))
    delta_x_star = np.zeros((x_size, T))
    
    P[:,:,-1] = QT
    p[:,-1] = qT
    delta_x_star[:,0] = x0
    
    # Solve Riccati equation
    for t in reversed(range(T-1)):
        if t % 100 == 0:
            # Debug: Use the following string as a break point 
            aswfasfasfa = 1
        print("Reverse iteration: ", t)
        Qt = Q_trajectory[:,:,t]
        Rt = R_trajectory[:,:,t]
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        St = S_trajectory[:,:,t]
        Pt_plus_1 = P[:,:,t+1]
        
        if augmented:
            qt = q_trajectory[:,t][:,None]
            rt = r_trajectory[:,t][:,None]
            pt_plus_1 = p[:, t+1][:,None]
        else:
            qt = np.zeros(x_size)
            rt = np.zeros(u_size)
            pt_plus_1 = np.zeros(x_size)

        # print("\n\n\n")
        # print("Rt: ", Rt)
        # print("Bt: ", Bt)
        # print("Pt_plus_1: ", Pt_plus_1)
        
        # to_invert = Rt + Bt.T @ Pt_plus_1 @ Bt
        
        # if np.all(to_invert == cache):
        #     Mt_inv = cache_inv
        # else: 
        #     Mt_inv = np.linalg.inv(to_invert)
        #     cache = to_invert
        #     cache_inv = Mt_inv

        Mt_inv = np.linalg.inv(Rt + Bt.T @ Pt_plus_1 @ Bt)
        mt = rt + Bt.T @ pt_plus_1
        
        # before calculating Pt verify that no element in the Pt matrix is NaN
        if np.isnan( At.T @ Pt_plus_1 @ At - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St) + Qt).any():
            # There is an issue with the matrix
            # This happens when
            #    dfx[2, 0] = 1  # line 101 dynamics.py
            #    dfx[3, 1] = 1  # line 102 dynamics.py
            # are set to 1. If they are set to 0, the matrix is fine
            breakpoint()
            
            
        Pt = At.T @ Pt_plus_1 @ At - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St) + Qt
        pt = At.T @ pt_plus_1 - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ mt + qt

        P[:,:,t] = Pt
        p[:,t] = pt.squeeze()

    # Find K_star and sigma_star    
    for t in range(T-1):
        print("Forward iteration: ", t)
        Rt = R_trajectory[:,:,t]
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        St = S_trajectory[:,:,t]
        Pt_plus_1 = P[:,:,t+1]
        
        if augmented:
            rt = r_trajectory[:,t][:,None]
            pt_plus_1 = p[:,t+1][:,None]
        else:
            rt = np.zeros(u_size)
            pt_plus_1 = np.zeros(x_size)
        
        # to_invert = Rt + Bt.T @ Pt_plus_1 @ Bt
        
        # if np.all(to_invert == cache):
        #     Mt_inv = cache_inv
        # else:
        #     Mt_inv = np.linalg.inv(to_invert)
        #     cache = to_invert
        #     cache_inv = Mt_inv

        Mt_inv = np.linalg.inv(Rt + Bt.T @ Pt_plus_1 @ Bt)
        mt = rt + Bt.T @ pt_plus_1

        # TODO: add a regularization step here

        K_star[:,:,t] = -Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St)
        sigma_t = -Mt_inv @ mt

        sigma_star[:,t] = sigma_t.squeeze()

    # TODO: I think this loop can be unified with the previous one
    for t in range(T-1):
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        
        # Trajectory
        delta_u_star[:, t] = K_star[:,:,t] @ delta_x_star[:, t] + sigma_star[:,t]
        delta_x_star[:,t+1] = At @ delta_x_star[:,t] + Bt @ delta_u_star[:, t]

    return K_star, sigma_star, delta_x_star


def compute_LQR_trajectory(x_trajectory, u_trajectory, step_size = 0.1, max_iter = 100):
    
    x_size = x_trajectory.shape[0]
    u_size = u_trajectory.shape[0]
    T = x_trajectory.shape[1]

    A_trajectory = np.zeros((x_size, x_size, T))
    B_trajectory = np.zeros((u_size, u_size, T))
    Q_trajectory = np.zeros((x_size, x_size, T))
    R_trajectory = np.zeros((u_size, u_size, T))
    S_trajectory = np.zeros((u_size, x_size, T))
    q_trajectory = np.zeros((x_size, T))
    r_trajectory = np.zeros((u_size, T))
    
    x0 = x_trajectory[:,0]
    u0 = u_trajectory[:,0]

    x_opt_trajectory = np.tile(x0[:, np.newaxis, np.newaxis], (1, T, max_iter+1))
    u_opt_trajectory = np.tile(u0[:, np.newaxis, np.newaxis], (1, T, max_iter+1))
    
    
    
    Q_trajectory[:,:,-1] = cost.hessian_terminal_cost()
    
    q_trajectory[:,-1] = cost.grad_terminal_cost(x_opt_trajectory[:,T-1,0], x_trajectory[:,T-1])
    
    for k in range(max_iter):          
        for t in range(T):   
            print("Iteration: ", k, " Time step: ", t)
            
            dtheta1 = x_trajectory[0,t]
            dtheta2 = x_trajectory[1,t]
            theta1 = x_trajectory[2,t]
            theta2 = x_trajectory[3,t]
            tau1 = u_trajectory[0,t]
            
            dfx = dyn.jacobian_x_dot_wrt_x(dtheta1, dtheta2, theta1, theta2, tau1)
            dfu = dyn.jacobian_x_dot_wrt_u(theta2)
            
            A_trajectory[:,:,t] = dfx.T
            B_trajectory[:,:,t] = dfu.T

        for t in range(T-1):
            Q_trajectory[:,:,t] = cost.hessian1_J()
            R_trajectory[:,:,t] = cost.hessian2_J()
        
            q_trajectory[:,t] = cost.grad1_J(x_opt_trajectory[:,t,k], x_trajectory[:,t])
            r_trajectory[:,t] = cost.grad2_J(u_opt_trajectory[:,t,k], u_trajectory[:,t])
            
        K, sigma, delta_x = solve_ltv_LQR(x0,
                                          A_trajectory,
                                          B_trajectory, 
                                          Q_trajectory, 
                                          R_trajectory, 
                                          S_trajectory, 
                                          q_trajectory, 
                                          r_trajectory)
        
        for t in range(T-1):    
            u_opt_trajectory[:,t,k+1] = u_opt_trajectory[:,t,k] + step_size * (sigma[:,t] + K[:,:,t] @ delta_x[:,t]) \
                + K[:,:,t] @ (x_opt_trajectory[:, t, k+1]-x_opt_trajectory[:,t,k]- step_size * delta_x[:,t]) 
                
            x_opt_trajectory[:,t+1,k+1] = dyn.dynamics(x_opt_trajectory[:,t,k+1][:, None], 
                                        u_opt_trajectory[:,t,k+1][:, None], dt=params.dt)[0].flatten()
                        
            # print("K[:,:,t]: ", K[:,:,t])
            # print("sigma[:,t]: ", sigma[:,t])
            # print("delta_x[:,t]: ", delta_x[:,t])
            
            print(f"u_opt_trajectory:{u_opt_trajectory[:,t,k+1]} \nx_opt_trajectory:{x_opt_trajectory[:,t+1,k+1]}")
        
        x_trajectory = x_opt_trajectory[:,:,-1]
        u_trajectory = u_opt_trajectory[:,:,-1]
            
    return x_trajectory, u_trajectory