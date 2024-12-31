import numpy as np
import dynamics as dyn
import cost
import parameters as pm
import matplotlib.pyplot as plt
from numpy.linalg import inv

def NewtonForOPTCON(x_reference, u_reference):
    
    TT = x_reference.shape[1]
    max_iterations = 10000
    
    l = np.zeros(max_iterations)
    x_initial_guess = x_reference[:,0]
    u_initial_guess = u_reference[:,0]

    x_optimal = np.zeros((4,TT, max_iterations))
    u_optimal = np.zeros((4,TT, max_iterations))
    delta_u = np.zeros((4,TT, max_iterations))

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
     

    # Applichiamo il metodo di Newton per calcolare la traiettoria ottimale
    for k in range(max_iterations):
        
        # Calcoliamo il costo della traiettoria di partenza. Alla fine di questo ciclo, 
        #   lo confrontiamo con il costo della traiettoria calcolata durante il ciclo.
        l[k] = cost.J_Function(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, "LQR")
        print("\nIteration: ", k, " Cost: ", l[k])

        
        for t in range(TT-1):
            A[:,:,t] = dyn.jacobian_x_dot_wrt_x_simplified(x_optimal[:,t,k], u_optimal[:,t,k])
            B[:,:,t] = dyn.jacobian_x_dot_wrt_u_simplified(x_optimal[:,t,k])
            qt[:,t] = cost.grad1_J(x_optimal[:,t,k], x_reference[:,t])
            rt[:,t] = cost.grad2_J(u_optimal[:,t,k], u_reference[:,t])
        
        qT = cost.grad_terminal_cost(x_optimal[:,TT-1,k], x_reference[:,TT-1])
        Lambda[:,TT-1] = qT

        ########## Solve the costate equation [S20C5]
        for t in reversed(range(TT-1)):
            Lambda[:,t] = A[:,:,t].T @ Lambda[:,t+1] + qt[:,t]

        ########## Compute the descent direction [S8C9]
        for t in range(TT-1):
            Qt_Star[:,:,t] = cost.hessian1_J()\
                + dyn.hessian_x_dot_wrt_x_simplified(x_optimal[:,t,k], u_optimal[:,t,k]) @ Lambda[:,t+1]              
            
            Rt_Star[:,:,t] = cost.hessian2_J()\
                + dyn.hessian_x_dot_wrt_u_simplified() @ Lambda[:,t+1]                 
            
            St_Star[:,:,t] = cost.hessian_12_J(x_optimal[:,t,k], u_optimal[:,t,k])\
                + dyn.hessian_x_dot_wrt_xu_simplified(x_optimal[:,t,k]) @ Lambda[:,t+1]
            
        QT_Star = cost.hessian_terminal_cost()


        ########## Compute the optimal control input [S18C9]
        # To compute the descent direction, the affine LQR must be solved
        K_star, sigma_star =  Affine_LQR_solver(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, A, B, Lambda, Qt_Star, Rt_Star, St_Star, QT_Star, qt, rt, qT)

        for t in range(1, TT-1):
            gamma = 0.01 # TODO: Gamma dovrebbe essere calcolato con Armijo
            u_optimal[:,t, k+1] = u_optimal[:,t, k] + K_star[:,:,t] @ (x_optimal[:,t, k+1] - x_optimal[:,t,k]) + gamma*sigma_star[:,t]
            x_optimal[:,t+1, k+1] = dyn.dynamics(x_optimal[:,t,k+1], u_optimal[:,t,k+1])[0]

        l[k+1] = cost.J_Function(x_optimal[:,:,k+1], u_optimal[:,:,k+1], x_reference, u_reference, "LQR")

        if l[k+1] - l[k] < 1e-6:
            break
    
    return x_optimal[:,:,k], u_optimal[:,:,k], l[:k]
    







        

                                     

        

        
        





def Affine_LQR_solver(x_optimal, u_optimal, x_reference, u_reference, At, Bt, Lambda, Qt_Star, Rt_Star, St_Star, QT_Star, qt, rt, qT):
    TT = x_reference.shape[1]
    max_iterations = 100
    Qt_tilde = np.zeros((5,5,TT))
    Rt_tilde = np.zeros((5,5,TT))
    St_tilde = np.zeros((5,5,TT))
    delta_x = np.zeros((4,TT))
    delta_u = np.zeros((4,TT))
    delta_x_tilde = np.zeros((5,TT))
    A_tilde = np.zeros((5,5,TT))
    B_tilde = np.zeros((5,5,TT))
    ct = np.zeros((4,1)) 
    Pt = np.zeros((5,5,TT))
    pt = np.zeros((5,TT))

    Kt = np.zeros((4,4,TT-1))
    sigma_t = np.zeros((4,TT-1))
    

    ################# Potrebbe non servire #################
    #for t in range(TT):                                  ##
    #    delta_x[:,t] = x_optimal[:,t]-x_reference[:,t]   ##
    #    delta_u[:,t] = u_optimal[:,t]-u_reference[:,t]   ##
    #    delta_x_tilde[:,t]= np.block([   [1],            ##
    #                                 [delta_x[:,t]]])    ##
    ########################################################

    for t in range(TT-1):
        Qt_tilde[:,:,t] = np.block([[      0,      qt[:,t].T], 
                                    [qt[:,t], Qt_Star[:,:,t]]])
        
        St_tilde[:,:,t] = np.block([rt[:,t], St_Star[:,:,t]])

        Rt_tilde[:,:,t] = Rt_Star[:,:,t]
        # TODO: Provare a considerare il sistema come se dotato di una dinamica affine e non come 
        # un sistema lineare (ovvero inserire il termine c di [S15C9], che per noi potrebbe essere "- M_inv_ext @ (C_ext + G_ext)")
        A_tilde[:,:,t] = np.block([[ 1,   np.zeros((1,4))], 
                                   [ct,          At[:,:,t]]])

        B_tilde[:,:,t] = np.block([[np.zeros((1,4))],
                                   [       Bt[:,:,t]]])
    # TODO:Non sono certo che la definizione del terminal cost dell'affine LQR sia corretta        
    Qt_tilde[:,:,TT-1] = np.block([[np.zeros((1,1)), qT.T], 
                                   [qT,           QT_Star]])

    ######### Solve the augmented system Riccati Equation [S16C9]
    Pt[:,:,TT-1] = Qt_tilde[:,:,TT-1]
    pt[:,TT-1] = qT

    for t in reversed(range(TT-1)):
       # Assegna ciascun valore ad una variabile temporanea per rendere l'equazione comprensibile
       A = A_tilde[:,:,t]
       B = B_tilde[:,:,t]
       Q = Qt_tilde[:,:,t]
       R = Rt_tilde[:,:,t]
       S = St_tilde[:,:,t]
       r = rt[:,t]
       q = qt[:,t]
       

       Kt[:,:,t]=-inv(R + B.T @ Pt[:,:,t+1] @ B) @ (S @ B.T @ Pt[:,:,t+1] @ A)
       sigma_t[:,t]=-inv(R + B.T @ Pt[:,:,t+1] @ B) @ (r + B.T @ pt[:,t+1] + B.T @ Pt[:,:,t+1] @ ct)
       pt[:,t] = q + A.T @ pt[:,t+1] + A.T @ Pt[:,:,t+1] @ ct - Kt[:,:,t].T @ (R + B.T @ Pt[:,:,t+1] @ B) @ sigma_t[:,t]
       Pt[:,:,t] = Q + A.T @ Pt[:,:,t+1] @ A - Kt[:,:,t].T @ (R + B.T @ Pt[:,:,t+1] @ B) @ Kt

    return Kt, sigma_t