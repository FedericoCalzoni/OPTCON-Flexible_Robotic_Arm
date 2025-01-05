import numpy as np
import matplotlib.pyplot as plt
from parameters import beta, c
import cost
import dynamics as dyn



def armijo_v2(x_init, x_reference, u_init, u_reference, delta_u, gradJ, J, Kt, sigma_t, step_size_0=0.1, max_iterations=10):
    # Estraggo le dimensioni di x ed u (4 e 4) ed il numero di iterazioni in una traiettoria (t_sim) 
    x_size = x_reference.shape[0]
    horizon = x_reference.shape[1]

    # Definisco la risoluzione nei plot; ogni curva è composta da 50 punti
    resolution = 25   

    ## Inizializzo le seguenti variabili:
    #  -    step_size: gamma che si prende in esame all'iterazione i-esima
    #  -    gamma: da 1 a resolution, serve per generare i plot
    #  -    step_sizes: vettore di tutti gli step_size valutati fino all'iterazione i-esima
    #  -    costs_armijo: vettore dei costi della funzione valutati fino all'iterazione i-esima 
    step_size = step_size_0
    gamma = np.linspace(0, 1, resolution)
    step_sizes = []
    costs_armijo = []

    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]

    x_temp = np.zeros((x_size,horizon))
    u_temp = np.zeros((u_size,horizon))

    #descent = q_trajectory.ravel() @ delta_u.ravel()
    #descent = np.zeros((horizon-1,1))
    #for t in range(horizon-1):
    #    descent[t] = gradJ[:,t] * delta_u[:,t]

    descent = 0
    for t in range(horizon-1):
        descent = descent + gradJ[:,t] * delta_u[:,t]


    for k in range(max_iterations):
        x_temp[:,0] = x_init[:,0].flatten()
        u_temp[:,0] = u_init[:,0].flatten()
        for t in range(horizon-1):
            u_temp[:,t] = u_temp[:,t] + Kt[:,:,t] @ (x_temp[:,t] - x_reference[:,t]) + sigma_t[:,t] * step_size
            x_temp[:,t+1] = dyn.dynamics(x_temp[:,t].reshape(-1, 1), u_temp[:,t].reshape(-1, 1))

        J_temp = cost.J_Function(x_temp, u_temp, x_reference, u_reference, "LQR")

        step_sizes.append(step_size)
        costs_armijo.append(J_temp)

        if (J_temp > J + c * step_size * descent):
            #print('J_temp = {}'.format(J_temp))
            step_size = beta * step_size

        else:
            print(f'Selected Armijo step_size = {step_size}')
            break

        #print(f'Step size at iter {k+1} = {step_size}')
    
    # Armijo Plot
    x_temp_sec = np.zeros((x_size, horizon, resolution))
    u_temp_sec = np.zeros((u_size, horizon-1, resolution))

    x_temp_sec[:,0,0] = x_init[:,0].flatten()
    u_temp_sec[:,0,0] = u_init[:,0].flatten()

    J_plot = np.zeros(resolution)
    for j in range(resolution):
        x_temp_sec[:,0, j] = x_init[:,0].flatten()
        u_temp_sec[:,0, j] = u_init[:,0].flatten()

        for t in range(horizon-1):
            u_temp_sec[:,t,j] = u_temp_sec[:,t,j] + Kt[:,:,t] @ (x_temp_sec[:,t,j] - x_reference[:,t]) + sigma_t[:,t] * gamma[j]
            x_temp_sec[:,t+1,j] = dyn.dynamics(x_temp_sec[:,t,j].reshape(-1, 1), u_temp_sec[:,t,j].reshape(-1, 1))
            
        J_plot[j] = cost.J_Function(x_temp_sec[:,:,j], u_temp_sec[:,:,j], x_reference, u_reference, "LQR")
    

    
    plt.plot(gamma, J+c*gamma*descent, color='red', label='Armijo Condition')
    plt.plot(gamma, J+gamma*descent, color='black', label='Tangent Line')
    plt.plot(gamma, J_plot, label='Cost Evolution')
    plt.scatter(step_sizes, costs_armijo, color='blue', label='Armijo Steps')
    plt.legend()
    plt.show()



    return step_size