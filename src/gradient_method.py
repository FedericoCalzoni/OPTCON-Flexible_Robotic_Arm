import dynamics as dyn
import cost 
import numpy as np
import armijo


def gradient_method(x_init, u_init, x_des, u_des, max_iterations=100, tol=1e-5):

    T = x_des.shape[1]
    
    x_temp = np.zeros((x_init.shape[0], T))
    u_temp = np.zeros((u_init.shape[0], T))
    
    x_temp = x_init
    u_temp = u_init
    
    x_temp[:,0] = x_des[:,0]
        
    J_history = np.zeros(max_iterations)
    lmbd_traj = np.zeros((x_init.shape[0], T))
    
    for k in range(max_iterations):
        
        # Step 1: Compute descent direction
        
        J_history[k] = cost.J_Function(x_temp, u_temp, x_des, u_des, "LQR")
        
        # costate terminal condition
        lmbd_traj[:,T-1] = cost.grad_terminal_cost(x_temp[:,T-1], x_des[:,T-1]) 
        
        
        for t in reversed(range(T-1)):
            q_t = cost.grad1_J(x_temp[:,t], x_des[:,t])
            r_t = cost.grad2_J(u_temp[:,t], u_des[:,t])
            dfx = dyn.jacobian_x_dot_wrt_x(x_temp[:,t], u_temp[:,t])
            dfu = dyn.jacobian_x_dot_wrt_u(x_temp[:,t])
                        
            At = dfx.T # maybe not transpose
            Bt = dfu            
            
            lmbd_traj[:,t]= q_t + At.T @ lmbd_traj[:,t+1] # costate update
            
            # descent direction
            grad_J_wrt_u = r_t + Bt.T @ lmbd_traj[:,t+1]
            delta_u = - grad_J_wrt_u # negative gradient
            
        # Step 2 & 3: Compute a new input and state trajectory
        
        # step_size = armijo.select_step_size(x_temp[:,0], # are we using the correct initial state?
        #                                     u_temp[:,0], 
        #                                     x_des, 
        #                                     u_des,
        #                                     J[k])
        
        step_size = 1
        
        for t in range(T-1):
            u_temp[:,t] += step_size * delta_u
            x_temp[:,t+1] = dyn.dynamics(x_temp[:,t], u_temp[:,t])[0]
            
        if k > 0 and abs(J_history[-1] - J_history[-2]) < tol:
            break
        
        print("x_temp: ", x_temp)
        print("u_temp: ", u_temp)
            
    return x_temp, u_temp, J_history, lmbd_traj
            
