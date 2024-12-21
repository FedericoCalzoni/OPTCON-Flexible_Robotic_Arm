import dynamics as dyn
import cost 
import numpy as np
import armijo


def gradient_method(x_init, u_init, x_reference, u_reference, max_iterations=100):

    T = x_reference.shape[1]
    
    x_temp = np.zeros((x_init.shape[0], T))
    u_temp = np.zeros((u_init.shape[0], T))
    
    x_temp = x_init
    u_temp = u_init
    
    J = np.zeros(max_iterations)
    lmbd = np.zeros((x_init.shape[0], T))
    
    for k in range(max_iterations-1):
        
        # Step 1: Compute descent direction
        
        J[k] = cost.J_Function(x_temp, u_temp, x_reference, u_reference)
        lmbd[:,T-1] = cost.grad_terminal_cost(x_temp[:,T-1], x_reference[:,T-1])
        
        
        for t in reversed(range(T-1)):
            q_t = cost.grad1_J(x_temp[:,t], x_reference[:,t])
            r_t = cost.grad2_J(u_temp[:,t], u_reference[:,t])
            dfx_ref, dfu_ref = dyn.dynamics(x_reference[:,t], u_reference[:,t])[1:]
            
            At = dfx_ref.T
            Bt = dfu_ref.T
            
            lmbd[:,t]=At.T@lmbd[:,t+1]+q_t # costate equation
            dJ_temp = Bt.T@lmbd[:,t+1]+r_t # gradient of J wrt u
            delta_u_temp = - dJ_temp # descent direction
            
        # Step 2 & 3: Compute a new input and state trajectory
        
        # step_size = armijo.select_step_size(x_temp[:,0], 
        #                                     u_temp[:,0], 
        #                                     x_reference, 
        #                                     u_reference,
        #                                     J[k])
        
        step_size = 0.01
        
        for t in range(T-1):
            u_temp[:,t] = u_temp[:,t] + step_size * delta_u_temp
            x_temp[:,t+1] = dyn.dynamics(x_temp[:,t][:,None], u_temp[:,t][:,None])[0].flatten()
            
    return x_temp, u_temp, J, lmbd
            
