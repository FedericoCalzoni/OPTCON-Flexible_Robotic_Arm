import numpy as np
import cost
import dynamics as dyn
from parameters import c, beta

def select_step_size(x_init, u_init, x_reference, u_reference, J, step_size_0=0.01, max_iterations=100):

    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]
        
    horizon = u_reference.shape[1]
    print('horizon = {}'.format(horizon))
    
    for i in range(max_iterations):
        
        step_sizes = []
        costs_armijo = []
        
        step_size=step_size_0
        
        x_temp = np.zeros((x_size,horizon+1))
        u_temp = np.zeros((u_size,horizon))

        x_temp[:,0] = x_init.flatten()
        u_temp[:,0] = u_init.flatten()
        
        for j in range(horizon):
            delta_u = - cost.grad2_J(u_temp[:,j], u_reference[:,j])
            u_temp[:,j] = u_temp[:,j] + step_size * delta_u
            x_temp[:,j+1] = dyn.dynamics(x_temp[:,j].reshape(-1, 1), 
                                         u_temp[:,j].reshape(-1, 1))[0].flatten()
        
        J_temp = cost.J_Function(x_temp, u_temp, x_reference, u_reference)
        
        step_sizes.append(step_size)
        costs_armijo.append(J_temp)
        
        if J_temp > J + c * step_size * delta_u.T @ delta_u:
            print('J_temp = {}'.format(J_temp))
            step_size = beta * step_size
            
        else:
            print('Armijo step_size = {}'.format(step_size))
            break
            
        print('step_size solution not found after {} iterations'.format(horizon))
            
    return step_size