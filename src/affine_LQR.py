import dynamics as dyn
import cost 
import numpy as np
import armijo

# TODO: Implement the affine_LQR function, now that we have the gradient_method function as a starting point.
def affine_LQR(x_init, u_init, x_reference, u_reference, max_iterations=100):

    T = x_reference.shape[1]
    
    x_temp = np.zeros((x_init.shape[0], T))
    u_temp = np.zeros((u_init.shape[0], T))
    
    x_temp = x_init
    u_temp = u_init
    
    J = np.zeros(max_iterations)
    lmbd = np.zeros((x_init.shape[0], T))
    
    for k in range(max_iterations):
        
        # Step 1: Compute descent direction
        
        J[k] = cost.J_Function(x_temp, u_temp, x_reference, u_reference)
        
        # costate terminal condition
        lmbd[:,T-1] = cost.grad_terminal_cost(x_temp[:,T-1], x_reference[:,T-1]) 
        
        # TODO: compute Q_t, R_t, S_t, and Q_T matrices
        
        for t in reversed(range(T-1)):
            # K_t_star = -(R_t + B_t^T P_{t+1} B_t)^-1 * (S_t + B_t^T P_{t+1} A_t)
            # sigma_t_star = -(R_t + B_t^T P_{t+1} B_t)^-1 * (r_t + B_t^T p_{t+1} + B_t^T P_{t+1} c_t)
            # p_t = q_t + A_t^T p_{t+1} + A_t^T P_{t+1} c_t - K_t_star^T (R_t + B_t^T P_{t+1} B_t) sigma_t_star
            # P_t = Q_t + A_t^T P_{t+1} A_t - K_t_star^T (R_t + B_t^T P_{t+1} B_t) K_t_star
            
            q_t = cost.grad1_J(x_temp[:,t], x_reference[:,t])
            r_t = cost.grad2_J(u_temp[:,t], u_reference[:,t])
            _, dfx_ref, dfu_ref = dyn.dynamics(x_reference[:,t], u_reference[:,t])
            
            At = dfx_ref.T
            Bt = dfu_ref.T
            
            lmbd[:,t]=At.T@lmbd[:,t+1]+q_t # costate update
            
            # descent direction
            dJ_temp = Bt.T@lmbd[:,t+1]+r_t # gradient of J wrt u
            delta_u = - dJ_temp # negative gradient
            
        # Step 2 & 3: Compute a new input and state trajectory
        
        # step_size = armijo.select_step_size(x_temp[:,0], # are we using the correct initial state?
        #                                     u_temp[:,0], 
        #                                     x_reference, 
        #                                     u_reference,
        #                                     J[k])
        
        step_size = 0.01
        
        for t in range(T-1):
            u_temp[:,t] = u_temp[:,t] + step_size * delta_u
            x_temp[:,t+1] = dyn.dynamics(x_temp[:,t][:,None], u_temp[:,t][:,None])[0].flatten()
            
    return x_temp, u_temp, J, lmbd
            
