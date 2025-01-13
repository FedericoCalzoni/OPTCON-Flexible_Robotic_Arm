import numpy as np
import matplotlib.pyplot as plt
from parameters import beta, c, Arm_plot, Arm_plot_every_k_iter, arm_max_iter
import cost
import dynamics as dyn

def armijo_v2(x_trajectory, x_reference, u_trajectory, u_reference, delta_u, gradJ, J, Kt, sigma_t, iteration, step_size_0=0.1):

    x_size = x_reference.shape[0]
    horizon = x_reference.shape[1]

    # resolution for plotting the cost function
    resolution = 25   
    
    ## Initialize the following variables:
    #  -    step_size: gamma that is considered at the i-th iteration
    #  -    gamma: from 1 to resolution, used to generate the plots
    #  -    step_sizes: vector of all the step_sizes evaluated until the i-th iteration
    #  -    costs_armijo: vector of the costs of the function evaluated until the i-th iteration
    
    step_size = step_size_0
    gamma = np.linspace(0, 1, resolution)
    step_sizes = []
    costs_armijo = []

    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]

    x_update = np.zeros((x_size,horizon))
    u_update = np.zeros((u_size,horizon))
    
    for i in range(arm_max_iter):
        x_update[:,:] = x_trajectory
        u_update[:,:] = u_trajectory

    #descent = q_trajectory.ravel() @ delta_u.ravel()
    #descent = np.zeros((horizon-1,1))
    #for t in range(horizon-1):
    #    descent[t] = gradJ[:,t] * delta_u[:,t]

    descent = 0
    for t in range(horizon-1):
        descent = descent + gradJ[:,t] * delta_u[:,t]

    for i in range(arm_max_iter-1):
        x_update[:,:] = x_trajectory
        u_update[:,:] = u_trajectory
        for t in range(horizon-1):
            u_update[:,t] = u_trajectory[:,t] + Kt[:,:,t] @ (x_update[:,t] - x_trajectory[:,t]) + sigma_t[:,t] * step_size
            x_update[:,t+1] = dyn.dynamics(x_update[:,t].reshape(-1, 1), u_update[:,t].reshape(-1, 1))

        J_temp = cost.J_Function(x_update, u_update, x_reference, u_reference, "LQR")

        step_sizes.append(step_size)
        costs_armijo.append(J_temp)

        if (J_temp > J + c * step_size * descent):
            #print('J_temp = {}'.format(J_temp))
            step_size = beta * step_size
            if i == arm_max_iter-2:
                print(f'Armijo method did not converge in {arm_max_iter} iterations')
                step_size = 0
                break

        else:
            print(f'Selected Armijo step_size = {step_size}')
            break

        #print(f'Step size at iter {k+1} = {step_size}')
    

    if Arm_plot == True and iteration%Arm_plot_every_k_iter == 0 and iteration!=0:
        # Armijo Plot
        x_temp_sec = np.zeros((x_size, horizon, resolution))
        u_temp_sec = np.zeros((u_size, horizon, resolution))
        J_plot = np.zeros(resolution)
   
        for j in range(resolution):
            x_temp_sec[:,:,j] = x_trajectory
            u_temp_sec[:,:,j] = u_trajectory
   
        for j in range(resolution):
            for t in range(horizon-1):
                u_temp_sec[:,t,j] = u_trajectory[:,t] + Kt[:,:,t] @ (x_temp_sec[:,t,j] - x_trajectory[:,t]) + sigma_t[:,t] * gamma[j]
                x_temp_sec[:,t+1,j] = dyn.dynamics(x_temp_sec[:,t,j].reshape(-1, 1), u_temp_sec[:,t,j].reshape(-1, 1))
                
            J_plot[j] = cost.J_Function(x_temp_sec[:,:,j], u_temp_sec[:,:,j], x_reference, u_reference, "LQR")



        plt.plot(gamma, J+c*gamma*descent, color='red', label='Armijo Condition')
        plt.plot(gamma, J+gamma*descent, color='black', label='Tangent Line')
        plt.plot(gamma, J_plot, color='green', label='Cost Evolution')
        plt.scatter(step_sizes, costs_armijo, color='blue', label='Armijo Steps')
        plt.grid()
        plt.legend()
        plt.show()


    return step_size