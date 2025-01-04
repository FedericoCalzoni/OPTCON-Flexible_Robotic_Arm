import numpy as np
import dynamics as dyn
import cost
import parameters as pm
import matplotlib.pyplot as plt
from numpy.linalg import inv
import armijotest as armijo

def NewtonForOPTCON(x_reference, u_reference):
    
    TT = x_reference.shape[1]
    max_iterations = 10000
    
    l = np.zeros(max_iterations)
    x_initial_guess = x_reference[:,0]
    u_initial_guess = u_reference[:,0]

    x_optimal = np.zeros((4,TT, max_iterations))
    u_optimal = np.zeros((4,TT, max_iterations))

    qt = np.zeros((4,TT-1))
    rt = np.zeros((4,TT-1))

    Lambda = np.zeros((4,TT))

    Qt_Star = np.zeros((4,4,TT-1))
    St_Star = np.zeros((4,4,TT-1))
    Rt_Star = np.zeros((4,4,TT-1))
    A = np.zeros((4,4,TT-1))
    B = np.zeros((4,4,TT-1))

    # Inizializzo la prima istanza della traiettoria ottimale
    for t in range(TT):
        x_optimal[:,t, 0] = x_initial_guess
        u_optimal[:,t, 0] = u_initial_guess
    
    gamma = 0.1

    # Applichiamo il metodo di Newton per calcolare la traiettoria ottimale
    for k in range(max_iterations):

        x_optimal[:,0, k] = x_initial_guess

        # Calcoliamo il costo della traiettoria di partenza. Alla fine di questo ciclo, 
        #   lo confrontiamo con il costo della traiettoria calcolata durante il ciclo.
        l[k] = cost.J_Function(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, "LQR")
        print("\nIteration: ", k, " Cost: ", l[k])

        
        for t in range(TT-1):
            A[:,:,t], B[:,:,t] = dyn.compute_jacobian_simplified(x_optimal[:,t,k], u_optimal[:,t,k])
            qt[:,t] = cost.grad1_J(x_optimal[:,t,k], x_reference[:,t])
            rt[:,t] = cost.grad2_J(u_optimal[:,t,k], u_reference[:,t])

        qT = cost.grad_terminal_cost(x_optimal[:,TT-1,k], x_reference[:,TT-1])
        # Lambda[:,TT-1] = qT

        ########## Solve the costate equation [S20C5]
        #for t in reversed(range(TT-1)):
        #    Lambda[:,t] = A[:,:,t].T @ Lambda[:,t+1] + qt[:,t]
        #    if(t%100 == 0):
        #        print("\nLambda: ", Lambda[:,t])

        ########## Compute the descent direction [S8C9]
        # Adopt Regularization methods
        for t in range(TT-1):
            Qt_Star[:,:,t] = cost.hessian1_J()           
            
            Rt_Star[:,:,t] = cost.hessian2_J()              
            
            St_Star[:,:,t] = cost.hessian_12_J(x_optimal[:,t,k], u_optimal[:,t,k])
            
        QT_Star = cost.hessian_terminal_cost()


        ########## Compute the optimal control input [S18C9]
        # To compute the descent direction, the affine LQR must be solved
        K_star, sigma_star, delta_u =  Affine_LQR_solver(x_optimal[:,:,k], x_reference, A, B, Qt_Star, Rt_Star, St_Star, QT_Star, qt, rt, qT)

        x_optimal[:,1, k+1] = x_initial_guess

        gamma = armijo.armijo_v2(x_optimal[:,:,k], x_reference, u_optimal[:,:,k], u_reference, delta_u, qt, qT, l[k], K_star, sigma_star, step_size_0=1)

        for t in range(1, TT-1): 
            u_optimal[:,t, k+1] = u_optimal[:,t, k] + K_star[:,:,t] @ (x_optimal[:,t, k+1] - x_optimal[:,t,k]) + gamma*sigma_star[:,t]
            x_optimal[:,t+1, k+1] = dyn.dynamics(x_optimal[:,t,k+1], u_optimal[:,t,k+1])[0].flatten()

        l[k+1] = cost.J_Function(x_optimal[:,:,k+1], u_optimal[:,:,k+1], x_reference, u_reference, "LQR")

        if np.abs(l[k+1] - l[k]) < 1e-6:
            break
    
    return x_optimal[:,:,k], u_optimal[:,:,k], l[:k]


def Affine_LQR_solver(x_optimal, x_reference, A, B, Qt_Star, Rt_Star, St_Star, QT_Star, qt, rt, qT):
    TT = x_reference.shape[1]
    
    delta_x = np.zeros((4,TT))
    delta_u = np.zeros((4,TT))

    delta_x[:,0] = x_optimal[:,0] - x_reference[:,0]

    ct = np.zeros((4,1)) 
    Pt = np.zeros((4,4,TT))
    pt = np.zeros((4,TT))
    
    Kt = np.zeros((4,4,TT-1))
    sigma_t = np.zeros((4,TT-1))

    ######### Solve the augmented system Riccati Equation [S16C9]
    Pt[:,:,TT-1] = QT_Star
    pt[:,TT-1] = qT

    for t in reversed(range(TT-1)):
       # Assegna ciascun valore ad una variabile temporanea per rendere l'equazione comprensibile
       At = A[:,:,t]
       Bt = B[:,:,t]
       Qt = Qt_Star[:,:,t]
       Rt = Rt_Star[:,:,t]
       St = St_Star[:,:,t]
       r = rt[:,t].reshape(-1, 1)
       q = qt[:,t].reshape(-1, 1)
       

       Kt[:,:,t]=-inv(Rt + Bt.T @ Pt[:,:,t+1] @ Bt) @ (St @ Bt.T @ Pt[:,:,t+1] @ At)
       sigma_t[:,t]=-inv(Rt + Bt.T @ Pt[:,:,t+1] @ Bt) @ (r + Bt.T @ pt[:,t+1].reshape(-1, 1) + Bt.T @ Pt[:,:,t+1] @ ct).flatten()
       pt[:,t] = (q + At.T @ pt[:,t+1].reshape(-1, 1) + At.T @ Pt[:,:,t+1] @ ct - Kt[:,:,t].T @ (Rt + Bt.T @ Pt[:,:,t+1] @ Bt) @ sigma_t[:,t].reshape(-1, 1)).flatten()
       Pt[:,:,t] = Qt + At.T @ Pt[:,:,t+1] @ At - Kt[:,:,t].T @ (Rt + Bt.T @ Pt[:,:,t+1] @ Bt) @ Kt[:,:,t]


    for t in range(1, TT-1):
        delta_u[:,t] = Kt[:,:,t] @ (x_optimal[:,t] - x_reference[:,t]) + sigma_t[:,t]
        delta_x[:,t+1] = A[:,:,t] @ delta_x[:,t-1] + B[:,:,t] @ delta_u[:,t]


    return Kt, sigma_t, delta_u