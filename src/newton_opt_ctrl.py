import numpy as np
import dynamics as dyn
import cost
import parameters as pm
import matplotlib.pyplot as plt
from numpy.linalg import inv
import armijotest as armijo

def newton_for_optcon(x_reference, u_reference):
    print('\t   -----------------------------------------\n \
          Starting Newton Method for Optimal Control\n \
          ----------------------------------------- ')
    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]
    TT = x_reference.shape[1]
    max_iterations = 1000
    
    l = np.zeros(max_iterations) # Cost function
    x_initial_guess = x_reference[:,0]
    u_initial_guess = u_reference[:,0]

    x_optimal = np.zeros((x_size, TT, max_iterations))
    u_optimal = np.zeros((u_size, TT, max_iterations))

    qt = np.zeros((x_size,TT-1))
    rt = np.zeros((u_size,TT-1))

    Lambda = np.zeros((4,TT))
    GradJ_u = np.zeros((u_size,TT-1))

    Qt_Star = np.zeros((x_size,x_size,TT-1))
    St_Star = np.zeros((u_size, x_size,TT-1))
    Rt_Star = np.zeros((u_size, u_size,TT-1))

    K_Star = np.zeros((u_size, x_size, TT-1, max_iterations))
    sigma_star = np.zeros((u_size, TT-1, max_iterations))
    delta_u = np.zeros((u_size, TT-1, max_iterations))

    A = np.zeros((x_size,x_size,TT-1))
    B = np.zeros((x_size,u_size,TT-1))

    newton_finished = False

    # Inizializzo la prima istanza della traiettoria ottimale
    for t in range(TT):
        x_optimal[:,t, 0] = x_initial_guess
        u_optimal[:,t, 0] = u_initial_guess

    # Applichiamo il metodo di Newton per calcolare la traiettoria ottimale
    for k in range(max_iterations):
        first_plot_set(k, x_optimal, x_reference, u_optimal, u_reference, newton_finished)
        
        l[k] = cost.J_Function(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, "LQR")

        
        if k == 0:
            print(f"\nIteration: {k} \tCost: {l[k]}")

        else: 
            # Controllo se la terminal condition è soddisfatta
            relative_cost_reduction = np.abs(l[k] - l[k-1])/l[k-1]
            print(f"\nIteration: {k} \tCost: {l[k]}\tCost reduction: {l[k] - l[k-1]}\tRelative cost reduction:{relative_cost_reduction}")
            
            if relative_cost_reduction < 1e-10:
                break
        
        
        
        # Inizializzo la x0 della prossima iterazione 
        x_optimal[:,0, k+1] = x_initial_guess
        
        # Calcolo le jacobiane di dinamica e costo
        for t in range(TT-1):
            A[:,:,t] = dyn.jacobian_x_new_wrt_x(x_optimal[:,t,k], u_optimal[:,t,k])
            B[:,:,t] = dyn.jacobian_x_new_wrt_u(x_optimal[:,t,k])
            qt[:,t] = cost.grad1_J(x_optimal[:,t,k], x_reference[:,t])
            rt[:,t] = cost.grad2_J(u_optimal[:,t,k], u_reference[:,t])
        qT = cost.grad_terminal_cost(x_optimal[:,-1,k], x_reference[:,-1])
        
        ########## Solve the costate equation [S20C5]
        # Compute the effects of the inputs evolution on cost (rt)
        # and on dynamics (B*Lambda)
        Lambda[:,-1] = qT
        for t in reversed(range(TT-1)):
            Lambda[:,t] = A[:,:,t].T @ Lambda[:,t+1] + qt[:,t]
            GradJ_u[:,t] = B[:,:,t].T @ Lambda[:,t+1] + rt[:,t]

        ########## Compute the descent direction [S8C9]
        # Adopt Regularization methods
        if k < 3:
            for t in range(TT-1):
                Qt_Star[:,:,t] = cost.hessian1_J()           
                Rt_Star[:,:,t] = cost.hessian2_J()              
                St_Star[:,:,t] = cost.hessian_12_J(x_optimal[:,t,k], u_optimal[:,t,k])  
            QT_Star = cost.hessian_terminal_cost()
        else :
            for t in range(TT-1):
                Qt_Star[:,:,t] = cost.hessian1_J() #+ dyn.hessian_x_new_wrt_x(x_optimal[:,t,k], u_optimal[:,t,k])@Lambda[:,t+1]         
                Rt_Star[:,:,t] = cost.hessian2_J() #+ dyn.hessian_x_new_wrt_u()@Lambda[:,t+1]              
                St_Star[:,:,t] = cost.hessian_12_J(x_optimal[:,t,k], u_optimal[:,t,k]);  
            QT_Star = cost.hessian_terminal_cost()


        ########## Compute the optimal control input [S18C9]
        # To compute the descent direction, the affine LQR must be solved
        K_Star[:,:,:,k], sigma_star[:,:,k], delta_u[:,:,k] =  \
            Affine_LQR_solver(x_optimal[:,:,k], x_reference, A, B, \
                              Qt_Star, Rt_Star, St_Star, QT_Star, qt, rt, qT)
        # Plot function outputs
        second_plot_set(k, sigma_star, delta_u, K_Star)
        
        # Compute the proper stepsize
        #if k == 5:
            #breakpoint()
        gamma = armijo.armijo_v2(x_optimal[:,:,k], x_reference, u_optimal[:,:,k], u_reference, delta_u[:,:,k], GradJ_u, l[k], K_Star[:,:,:,k], sigma_star[:,:,k], k, step_size_0=1)

        #gamma = 0.1

        for t in range(TT-1): 
            u_optimal[:,t, k+1] = u_optimal[:,t, k] + K_Star[:,:,t, k] @ (x_optimal[:,t, k+1] - x_optimal[:,t,k]) + gamma * sigma_star[:,t, k]
            x_optimal[:,t+1, k+1] = dyn.dynamics(x_optimal[:,t,k+1], u_optimal[:,t,k+1])

    print(f'Ho finito alla {k}^ iterazione')
    newton_finished = True
    first_plot_set(k, x_optimal, x_reference, u_optimal, u_reference, newton_finished)
    return x_optimal[:,:,k], u_optimal[:,:,k], l[:k]


def Affine_LQR_solver(x_optimal, x_reference, A, B, Qt_Star, Rt_Star, St_Star, QT_Star, q, r, qT):
    x_size = x_reference.shape[0]
    u_size = r.shape[0]
    TT = x_reference.shape[1]
    
    delta_x = np.zeros((x_size,TT))
    delta_u = np.zeros((u_size,TT-1))

    delta_x[:,0] = x_optimal[:,0] - x_reference[:,0]

    ct = np.zeros((x_size,1)) 
    P = np.zeros((x_size,x_size,TT))
    Pt = np.zeros((x_size,x_size))
    Ptt= np.zeros((x_size,x_size))

    p = np.zeros((x_size,TT))
    pt= np.zeros((x_size, 1))
    ptt=np.zeros((x_size, 1))
    
    K = np.zeros((u_size,x_size,TT-1))
    Kt= np.zeros((u_size,x_size))
    Sigma = np.zeros((u_size,TT-1))
    sigma_t = np.zeros((u_size, 1))

    ######### Solve the augmented system Riccati Equation [S16C9]
    P[:,:,TT-1] = QT_Star
    p[:,TT-1] = qT

    for t in reversed(range(TT-1)):
        # Assegna ciascun valore ad una variabile temporanea per rendere l'equazione comprensibile
        At  = A[:,:,t]
        Bt  = B[:,:,t]
        Qt  = Qt_Star[:,:,t]
        Rt  = Rt_Star[:,:,t]
        St  = St_Star[:,:,t]
        rt  = r[:,t].reshape(-1, 1)
        qt  = q[:,t].reshape(-1, 1)
        ptt = p[:,t+1].reshape(-1,1)
        Ptt = P[:,:,t+1]

        temp = (Rt + Bt.T @ Ptt @ Bt)
        inv_temp = inv(temp)

        Kt =-inv_temp @ (St + Bt.T @ Ptt @ At)
        sigma_t=-inv_temp @ (rt + Bt.T @ ptt + Bt.T @ Ptt @ ct)

        pt = qt + At.T @ ptt + At.T @ Ptt @ ct - Kt.T @ temp @ sigma_t
        Pt = Qt + At.T @ Ptt @ At - Kt.T @ temp @ Kt

        K[:,:,t] = Kt
        Sigma[:,t] = sigma_t
        p[:,t] = pt.flatten()
        P[:,:,t] = Pt 

    for t in range(TT-1):
        delta_u[:,t] = K[:,:,t] @ delta_x[:,t] + Sigma[:,t]
        delta_x[:,t+1] = A[:,:,t] @ delta_x[:,t] + B[:,:,t] @ delta_u[:,t]

    return K, Sigma, delta_u



def first_plot_set(k, x_optimal, x_reference, u_optimal, u_reference, newton_finished):
    x_size = x_optimal.shape[0]
    u_size = u_optimal.shape[0]
    if (pm.Newton_Optcon_Plots and k % pm.Newton_Plot_every_k_iterations== 0) \
        or (pm.plot_states_at_last_iteration and newton_finished):
            plt.figure()
            for i in range(x_size):
                plt.plot(x_optimal[i, :, k], color = 'blue', label =f'x_optimal[{i+1}]')
                plt.plot(x_reference[i,:], color = 'orange', label =f'x_reference[{i+1}]')
            plt.grid()
            plt.legend()
            plt.title(f'State Evolution\n$Iteration = {k}$')
            plt.show()

            plt.figure()
            for i in range(u_size):
                plt.plot(u_optimal[i,:,k], color = 'purple', label =f'u_optimal[{i+1}]')
                plt.plot(u_reference[i,:],color = 'yellow', label =f'u_reference[{i+1}]')
            plt.grid()
            plt.legend()
            plt.title(f'Input Evolution\n$Iteration = {k}$')
            plt.show()



def second_plot_set(k, sigma_star, delta_u, K_Star):
    if pm.Newton_Optcon_Plots and k % pm.Newton_Plot_every_k_iterations== 0:
            x_size = K_Star.shape[1]
            u_size = K_Star.shape[0]
            plt.figure()
            plt.title(f'Affine LQR solution at\nIteration {k}')
            for i in range(u_size):
                plt.plot(sigma_star[i,:, k], color = 'red', label = f'Sigma[{i}]')
                plt.plot(delta_u[i,:, k], color = 'purple', label = f'\Delta_u[{i}]')
                for j in range(x_size):
                    plt.plot(K_Star[i, j, :, k], color = 'blue', label = f'K[{i} , {j}]')
            plt.grid()
            plt.legend()
            plt.show()