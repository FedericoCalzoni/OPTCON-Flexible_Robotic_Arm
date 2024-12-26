import numpy as np
import dynamics as dyn
import cost
import parameters as params

def solve_ltv_LQR(A_trajectory, B_trajectory, Q_trajectory, R_trajectory, S_trajectory, x0_trajectory, q_trajectory, r_trajectory):
    
    cache = None
    cache_inv = None

    QT = Q_trajectory[:,:,-1]
    T = x0_trajectory.shape[1]
    qT = q_trajectory[:,-1]
    
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

    if q_trajectory is not None or r_trajectory is not None or qT is not None:
        print("Augmented term!")

    P = np.zeros((x_size, x_size, T))
    p = np.zeros((x_size, T))
    K_star = np.zeros((u_size, x_size, T))
    sigma_star = np.zeros((u_size, T))
    delta_u_star = np.zeros((u_size, T))
    delta_x_star = np.zeros((x_size, T))
    
    P[:,:,-1] = QT
    p[:,-1] = qT
    delta_x_star[:,0] = x0_trajectory[:,0]
    
    # Solve Riccati equation
    for t in reversed(range(T-1)):
        Qt = Q_trajectory[:,:,t]
        qt = q_trajectory[:,t][:,None]
        Rt = R_trajectory[:,:,t]
        rt = r_trajectory[:,t][:,None]
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        St = S_trajectory[:,:,t]
        Pt_plus_1 = P[:,:,t+1]
        pt_plus_1 = p[:, t+1][:,None]

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
        
        Pt = At.T @ Pt_plus_1 @ At - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St) + Qt
        pt = At.T @ pt_plus_1 - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ mt + qt

        P[:,:,t] = Pt
        p[:,t] = pt.squeeze()


    # Evaluate KK
    
    for t in range(T-1):
        Rt = R_trajectory[:,:,t]
        rt = r_trajectory[:,t][:,None]
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        St = S_trajectory[:,:,t]

        Pt_plus_1 = P[:,:,t+1]
        pt_plus_1 = p[:,t+1][:,None]
        
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

    for t in range(T-1):
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        
        # Trajectory
        delta_u_star[:, t] = K_star[:,:,t] @ delta_x_star[:, t] + sigma_star[:,t]
        delta_x_star[:,t+1] = At @ delta_x_star[:,t] + Bt @ delta_u_star[:, t]

    return K_star, sigma_star, delta_x_star


def compute_LQR_trajectory(A0, B0, x0_trajectory, u0_trajectory, x_reference, u_reference, step_size = 0.1, max_iter = 100):
    
    x_size = x0_trajectory.shape[0]
    u_size = u0_trajectory.shape[0]

    T = x0_trajectory.shape[1]
    
    # u_temp_trajectory[:,:,0] = u0_trajectory
    # x_temp_trajectory[:,:,0] = x0_trajectory
    # x_temp_trajectory[:,:,1] = x0_trajectory   
    x_temp_trajectory = x0_trajectory[:, :, np.newaxis] 
    x_temp_trajectory = np.repeat(x_temp_trajectory, max_iter+1, axis=2)
    u_temp_trajectory = u0_trajectory[:, :, np.newaxis] 
    u_temp_trajectory = np.repeat(u_temp_trajectory, max_iter+1, axis=2)

    A_trajectory = np.zeros((A0.shape[0], A0.shape[1], T))
    B_trajectory = np.zeros((B0.shape[0], B0.shape[1], T))
    Q_trajectory = np.zeros((x_size, x_size, T))
    R_trajectory = np.zeros((u_size, u_size, T))
    S_trajectory = np.zeros((u_size, x_size, T))
    q_trajectory = np.zeros((x_size, T))
    r_trajectory = np.zeros((u_size, T))
    
    # TODO: check if this is correct
    A_trajectory[:, :, :] = A0[:, :, np.newaxis]
    B_trajectory[:, :, :] = B0[:, :, np.newaxis] 
    
    Qt = cost.hessian1_J()
    Q_trajectory[:, :, :] = Qt[:, :, np.newaxis]
    Rt = cost.hessian2_J()
    R_trajectory[:, :, :] = Rt[:, :, np.newaxis]
    
    Q_trajectory[:,:,-1] = cost.hessian_terminal_cost()
    q_trajectory[:,-1] = cost.grad_terminal_cost(x_temp_trajectory[:,T-1,0], x_reference[:,T-2])
    
    for k in range(max_iter):          
        for t in range(T-1):   
            print("Iteration: ", k+1, " Time step: ", t)
            q_trajectory[:,t] = cost.grad1_J(x_temp_trajectory[:,t,k], x_reference[:,t])
            r_trajectory[:,t] = cost.grad2_J(u_temp_trajectory[:,t,k], u_reference[:,t])
            
            K, sigma, delta_x = solve_ltv_LQR(A_trajectory, 
                                              B_trajectory, 
                                              Q_trajectory, 
                                              R_trajectory, 
                                              S_trajectory, 
                                              x0_trajectory, 
                                              q_trajectory, 
                                              r_trajectory)
                
            u_temp_trajectory[:,t,k+1] = u_temp_trajectory[:,t,k] + step_size * (sigma[:,t] + K[:,:,t] @ delta_x[:,t]) \
                + K[:,:,t] @ (x_temp_trajectory[:, t, k+1]-x_temp_trajectory[:,t,k]- step_size * delta_x[:,t]) 
                
            # print("K[:,:,t]: ", K[:,:,t])
            # print("sigma[:,t]: ", sigma[:,t])
            print("delta_x[:,t]: ", delta_x[:,t])
            print("u_temp_trajectory[:,t,k+1]: ", u_temp_trajectory[:,t,k+1])
                
            x_new, dfx, dfu = dyn.dynamics(x_temp_trajectory[:,t,k+1][:, None], 
                                           u_temp_trajectory[:,t,k+1][:, None], dt=params.dt)
            
            x_temp_trajectory[:,t+1,k+1] = x_new.flatten()
            print("x_temp_trajectory[:,t+1,k+1]: ", x_temp_trajectory[:,t+1,k+1])
            A_trajectory[:,:,t+1] = dfx 
            B_trajectory[:,:,t+1] = dfu
        
    x_trajectory = x_temp_trajectory[:,:,-1]
    u_trajectory = u_temp_trajectory[:,:,-1]
            
    return x_trajectory, u_trajectory

# # Test
# Q0 = np.eye(4)
# R0 = np.eye(4)
# S0 = np.zeros((4, 4))
# QT = np.eye(4) * 10
# T = 100
# x0 = np.array([1, 0, 0, 0])
# u0 = np.zeros((4, 1))

# q0 = np.zeros((4, T))
# r0 = np.zeros((4, T))
# qT = np.zeros(4)

# x, u = compute_LQR_trajectory(A0, B0, Q0, R0, S0, QT, T, x0[:,None], q0, r0, qT, u0)
# print("x: ")
# print(x)
# print("u: ")
# print(u)

# # print shape of x and u
# print("Shape of x: ", x.shape)
# print("Shape of u: ", u.shape)