import numpy as np
import dynamics as dyn
import cost
import parameters as pm
import matplotlib.pyplot as plt
from numpy.linalg import inv
import armijotest as armijo

def Affine_LQR_solver(x_optimal, x_reference, A_trajectory, B_trajectory, Q_trajectory, R_trajectory, S_trajectory, q_trajectory, r_trajectory):
    
    x_size = x_reference.shape[0]
    u_size = r_trajectory.shape[0]
    T = x_reference.shape[1]
    
    P = np.zeros((x_size,x_size,T))
    K_star = np.zeros((u_size,x_size,T-1))
    sigma_star = np.zeros((u_size,T-1))
    delta_x_star = np.zeros((x_size,T))
    delta_u_star = np.zeros((u_size,T-1))
    
    P[:,:,-1] = Q_trajectory[:,:,-1]

    p = np.zeros((x_size,T))
    p[:,-1] = q_trajectory[:,-1]
    
    # DIFFERENT I use x0
    delta_x_star[:,0] = x_optimal[:,0] - x_reference[:,0]
    ct = np.zeros(x_size) 
    

    # Backward pass: Solve the augmented system Riccati Equation [S16C9]
    for t in reversed(range(T-1)):
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        Qt = Q_trajectory[:,:,t]
        Rt = R_trajectory[:,:,t]
        St = S_trajectory[:,:,t]
        
        q = q_trajectory[:,t]
        r = r_trajectory[:,t]
        
        M_inv = inv(Rt + Bt.T @ P[:,:,t+1] @ Bt)
        K_star[:,:,t]= - M_inv @ (St + Bt.T @ P[:,:,t+1] @ At)
        
        sigma_star[:,t] = - M_inv @ (r + Bt.T @ p[:,t+1] + Bt.T @ P[:,:,t+1] @ ct)
        
        p[:,t] = q + At.T @ p[:,t+1] + At.T @ P[:,:,t+1] @ ct - K_star[:,:,t].T @ (Rt + Bt.T @ P[:,:,t+1] @ Bt) @ sigma_star[:,t]
        #p[:, t] = q + At.T @ p[:,t+1] - (Bt.T @ P[:,:,t+1] @ At + St).T @ M_inv @ (r + Bt.T @ p[:,t+1])
        
        P[:,:,t] = Qt + At.T @ P[:,:,t+1] @ At - K_star[:,:,t].T @ (Rt + Bt.T @ P[:,:,t+1] @ Bt) @ K_star[:,:,t]
        # P[:,:,t] = Qt + At.T @ P[:,:,t+1] @ At - (Bt.T @ P[:,:,t+1] @ At + St).T @ M_inv @ (Bt.T @ P[:,:,t+1] @ At + St)
        # P[:,:,t] = Qt + At.T @ P[:,:,t+1] @ At - (At.T @ P[:,:,t+1] @ Bt) @ K_star[:,:,t]

    # Forward pass
    for t in range(1, T-1):
        At = A_trajectory[:,:,t]
        Bt = B_trajectory[:,:,t]
        
        
        delta_u_star[:,t] = K_star[:,:,t] @ (x_optimal[:,t] - x_reference[:,t]) + sigma_star[:,t]
        # delta_u_star[:,t] = K_star[:,:,t] @ delta_x_star[:, t] + sigma_star[:,t]
        
        delta_x_star[:,t+1] = At @ delta_x_star[:,t] + Bt @ delta_u_star[:,t]
        # delta_x_star[:,t+1] = At @ delta_x_star[:,t] + Bt @ delta_u_star[:, t]
        
    return K_star, sigma_star, delta_u_star

def newton_for_optcon(x_reference, u_reference, max_iterations=25):
    
    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]
    T = x_reference.shape[1]
    
    # DIFFERENT MATRICES SIZES
    A_trajectory = np.zeros((x_size,x_size,T-1))
    B_trajectory = np.zeros((x_size,u_size,T-1))
    Q_trajectory = np.zeros((x_size,x_size,T))
    R_trajectory = np.zeros((u_size,u_size,T-1))
    S_trajectory = np.zeros((u_size,x_size,T-1))
    q_trajectory = np.zeros((x_size,T))
    r_trajectory = np.zeros((u_size,T-1))
    
    x_initial_guess = x_reference[:,0]
    u_initial_guess = u_reference[:,0]

    # Initialize the optimal trajectory
    # SHOULD WE INITIALIZE AS INITIAL GUESS OR AS REFERENCE?
    x_optimal = np.zeros((x_size, T, max_iterations+1))
    u_optimal = np.zeros((u_size, T-1, max_iterations+1))
    
    for k in range(max_iterations + 1):
        for t in range(T):
            x_optimal[:, t, k] = x_initial_guess

    for k in range(max_iterations + 1):
        for t in range(T-1):
            u_optimal[:,t, k] = u_initial_guess
            
        
    # # initialize it instead as the reference trajectory  
    # x_optimal = np.repeat(x_reference[:, :, np.newaxis] , max_iterations+1, axis=2)
    # u_optimal = np.repeat(u_reference[:, :, np.newaxis] , max_iterations+1, axis=2)

    # DIFFERENT
    cost_func = np.zeros(max_iterations+1) # Cost function
    Lambda = np.zeros((4,T))
    GradJ_u = np.zeros((u_size,T-1))

    # Newton's method to solve the optimal control problem
    for k in range(max_iterations):

        # Compute the cost function for the current trajectory
        cost_func[k] = cost.J_Function(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, "LQR")
        
        for t in range(T-1):
            # Linearization of the dynamics
            A_trajectory[:,:,t] = dyn.jacobian_x_new_wrt_x(x_optimal[:,t,k], u_optimal[:,t,k])
            B_trajectory[:,:,t] = dyn.jacobian_x_new_wrt_u(x_optimal[:,t,k])
            

            ######### Compute the descent direction [S8C9]
            # Adopt Regularization methods 
            Q_trajectory[:,:,t] = cost.hessian1_J()
            R_trajectory[:,:,t] = cost.hessian2_J()
            S_trajectory[:,:,t] = cost.hessian_12_J(x_optimal[:,t,k], u_optimal[:,t,k])
            
            # Cost function derivatives
            q_trajectory[:,t] = cost.grad1_J(x_optimal[:,t,k], x_reference[:,t])
            r_trajectory[:,t] = cost.grad2_J(u_optimal[:,t,k], u_reference[:,t])

        # Terminal direction
        Q_trajectory[:,:,-1] = cost.hessian_terminal_cost()
        q_trajectory[:,-1] = cost.grad_terminal_cost(x_optimal[:,T-1,k], x_reference[:,T-1])
        
        Lambda[:,-1] = q_trajectory[:,-1] # Can this be put outside the loop?
        ########## Solve the costate equation [S20C5]
        # Compute the effects of the inputs evolution on cost (r)
        # and on dynamics (B*Lambda)
        for t in reversed(range(T-1)):
            Lambda[:,t] = A_trajectory[:,:,t].T @ Lambda[:,t+1] + q_trajectory[:,t]
            GradJ_u[:,t] = B_trajectory[:,:,t].T @ Lambda[:,t+1] + r_trajectory[:,t]
                   
        
        ########## Compute the optimal control input [S18C9]
        K_star, sigma_star, delta_u =  Affine_LQR_solver(x_optimal[:,:,k], # DIFFERENT
                                                         x_reference, # DIFFERENT
                                                         A_trajectory,
                                                         B_trajectory,
                                                         Q_trajectory,
                                                         R_trajectory,
                                                         S_trajectory,
                                                         q_trajectory,
                                                         r_trajectory)
        
        # PlotMe = True
        # if PlotMe == True:
        #     plt.figure()
        #     plt.title(f'Affine LQR solution at\nIteration {k}')
        #     for i in range(u_size):
        #         plt.plot(sigma_star[i,:], color = 'red', label = f'Sigma[{i}]')
        #         plt.plot(delta_u[i,:], color = 'purple', label = f'\Delta_u[{i}]')
        #         for j in range(x_size):
        #             plt.plot(K_star[i, j, :], color = 'blue', label = f'K[{i} , {j}]')
        #     plt.grid()
        #     plt.legend()
        #     plt.show()
            
        # if k%10 == 0 or k == max_iterations:   
        #     # plt.plot(K_star[0,0,:])
        #     # plt.plot(sigma_star[0,:])
        #     for i in range(4):
        #         plt.plot(x_optimal[i,:,k])
        #     plt.show()
 
        # step_size = armijo.armijo_v2(x_optimal[:,:,k], x_reference, u_optimal[:,:,k], u_reference, delta_u, GradJ_u, cost_func[k], K_star, sigma_star, k)
        # step_size = armijo.armijo_line_search(x_optimal[:,:,k], x_reference, u_optimal[:,:,k], u_reference, delta_u, GradJ_u, cost_func[k], K_star, sigma_star, k, step_size_0=1)
        step_size = 0.2
        
        # Compute the new trajectory using the optimal control input
        for t in range(T-1): 
            u_optimal[:,t,k+1] = u_optimal[:,t,k] + K_star[:,:,t] @ (x_optimal[:,t,k+1] - x_optimal[:,t,k]) + step_size * sigma_star[:,t]
            x_optimal[:,t+1,k+1] = dyn.dynamics(x_optimal[:,t,k+1],u_optimal[:,t,k+1])
        
        # Compare the cost of the trajectory computed during the current iteration with the cost of the previous iteration
        cost_func[k+1] = cost.J_Function(x_optimal[:,:,k+1], u_optimal[:,:,k+1], x_reference, u_reference, "LQR")
        print(f"\nIteration: {k} Cost: {cost_func[k+1]}   Cost reduction: {cost_func[k+1] - cost_func[k]}")
        if np.abs(cost_func[k+1] - cost_func[k]) < 1e-6:
            break
        
        print(f"u_opt_trajectory:{u_optimal[:,t,k+1]} \nx_opt_trajectory:{x_optimal[:,t+1,k+1]}")
        
        
    
    plt.figure()
    for i in range(4):
        plt.plot(x_optimal[i, :, k], color = 'blue', label =f'x_optimal[{i+1}]')
        plt.plot(x_reference[i,:], color = 'orange', label =f'x_reference[{i+1}]')
    for i in range(u_size):
        plt.plot(u_optimal[i,:,k], color = 'purple', label =f'u_optimal[{i+1}]')
        plt.plot(u_reference[i,:],color = 'yellow', label =f'u_reference[{i+1}]')
    plt.grid()
    plt.legend()
    plt.title(f'State Input Evolution\n$Iteration = {k}$')
    plt.show()

    return x_optimal[:,:,-1], u_optimal[:,:,-1]
