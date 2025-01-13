import numpy as np
import dynamics as dyn
import cost
import parameters as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import inv
import armijo as armijo
from visualizer import animate_double_pendulum as anim
import data_manager as dm
from LQR import LQR_system_regulator as LQR

def newton_for_optcon(x_reference, u_reference):
    print('\n\n\
    \t\t--------------------------------------------\n \
    \t\tLaunching: Newton Method for Optimal Control\n \
    \t\t--------------------------------------------')
    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]
    TT = x_reference.shape[1]
    max_iterations = 300
    
    l = np.zeros((max_iterations)) # Cost function
    # x_initial_guess = x_reference[:,0]
    # u_initial_guess = u_reference[:,0]

    x_optimal = np.zeros((x_size, TT, max_iterations+1))
    u_optimal = np.zeros((u_size, TT, max_iterations+1))

    qt = np.zeros((x_size,TT-1))
    rt = np.zeros((u_size,TT-1))

    Lambda = np.zeros((4,TT))
    GradJ_u = np.zeros((u_size,TT-1))

    Qt_Star = np.zeros((x_size, x_size,TT-1))
    St_Star = np.zeros((u_size, x_size,TT-1))
    Rt_Star = np.zeros((u_size, u_size,TT-1))

    K_Star = np.zeros((u_size, x_size, TT-1, max_iterations))
    sigma_star = np.zeros((u_size, TT-1, max_iterations))
    delta_u = np.zeros((u_size, TT-1, max_iterations))

    A = np.zeros((x_size,x_size,TT-1))
    B = np.zeros((x_size,u_size,TT-1))

    newton_finished = False

    # Initialize the first instance of the optimal trajectory
    # for t in range(TT):
    #     x_optimal[:,t, 0] = x_initial_guess
    #     u_optimal[:,t, 0] = u_initial_guess

    
    # if pm.initial_guess_given:
    #     x_optimal[:,:, 0], u_optimal[:,:, 0] = dm.load_optimal_trajectory('29')
    if  True:
        x_optimal[:,:, 0] = x_reference
        u_optimal[:,:, 0] = Initial_LQR(x_reference, u_reference)
    else:
        for t in range(TT):
            x_optimal[:,t, 0] = x_reference[:,t]
            u_optimal[:,t, 0] = u_reference[:,t]

    # Apply newton method to compute the optimal trajectory
    GradJ_u_history = []  # Store GradJ_u history
    delta_u_history = []  # Store delta_u history
    
    for k in range(max_iterations):
        first_plot_set(k, x_optimal, x_reference, u_optimal, u_reference, newton_finished)
        
        l[k] = cost.J_Function(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, "LQR")

        if k <= 1:
            print(f"\nIteration: {k} \tCost: {l[k]}")
        else:
            # Check if the terminal condition is satisfied
            relative_cost_reduction = np.abs(l[k] - l[k-1])/l[k-1]
            print(f"\nIteration: {k} \tCost: {l[k]} \tCost reduction: {l[k] - l[k-1]} \tRelative cost reduction: {relative_cost_reduction}")
            if relative_cost_reduction < 1e-10 or relative_cost_reduction > 1:
                break
        
        # # Gradient norm stopping criteria
        # if k == 0:
        #     print(f"\nIteration: {k} \tCost: {l[k]}")
        # else: 
        #     norm_GradJ_u = np.linalg.norm(GradJ_u)
        #     print(f"\nIteration: {k} \tCost: {l[k]}\tCost reduction: {l[k] - l[k-1]}\tGradient Norm: {norm_GradJ_u}")
            
        #     # Modified stopping criterion based on gradient norm
        #     if norm_GradJ_u < 1e-10:
        #         break
        
        # Initialization of x0 for the next iteration
        x_optimal[:,0, k+1] = x_reference[:, 0]
        
        # Compute the Jacobians of the dynamics and the cost
        for t in range(TT-1):
            A[:,:,t] = dyn.jacobian_x_new_wrt_x(x_optimal[:,t,k], u_optimal[:,t,k])
            B[:,:,t] = dyn.jacobian_x_new_wrt_u(x_optimal[:,t,k])
            qt[:,t] = cost.grad1_J(x_optimal[:,t,k], x_reference[:,t], t)
            rt[:,t] = cost.grad2_J(u_optimal[:,t,k], u_reference[:,t], t)
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
        for t in range(TT-1):
            Qt_Star[:,:,t] = cost.hessian1_J(t)           
            Rt_Star[:,:,t] = cost.hessian2_J(t)         
            St_Star[:,:,t] = cost.hessian_12_J(x_optimal[:,t,k], u_optimal[:,t,k])  
        QT_Star = cost.hessian_terminal_cost()


        ########## Compute the optimal control input [S18C9]
        # To compute the descent direction, the affine LQR must be solved
        K_Star[:,:,:,k], sigma_star[:,:,k], delta_u[:,:,k] =  \
            Affine_LQR_solver(x_optimal[:,:,k], x_reference, A, B, \
                              Qt_Star, Rt_Star, St_Star, QT_Star, qt, rt, qT)
            
        # Plot Affine_LQR outputs
        second_plot_set(k, sigma_star, delta_u, K_Star)
        
        # Compute the proper stepsize
        #if k == 5:
            #breakpoint()
        if k == 0:
            gamma = 0.01
        else:
            gamma = armijo.armijo_v2(x_optimal[:,:,k], x_reference, u_optimal[:,:,k], u_reference, delta_u[:,:,k], GradJ_u, l[k], K_Star[:,:,:,k], sigma_star[:,:,k], k, step_size_0=1)

        # gamma = 0.1

        for t in range(TT-1): 
            u_optimal[:,t, k+1] = u_optimal[:,t, k] + K_Star[:,:,t, k] @ (x_optimal[:,t, k+1] - x_optimal[:,t,k]) + gamma * sigma_star[:,t, k]
            x_optimal[:,t+1, k+1] = dyn.dynamics(x_optimal[:,t,k+1], u_optimal[:,t,k+1])

    print(f'Ho finito alla {k}^ iterazione')
    newton_finished = True
    first_plot_set(k, x_optimal, x_reference, u_optimal, u_reference, newton_finished)
    plot_optimal_intermediate_trajectory(x_reference, u_reference, x_optimal, u_optimal, k)
    return x_optimal[:,:,k], u_optimal[:,:,k], GradJ_u_history, delta_u_history

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

def Initial_LQR(x_ref, u_ref):
    print('\n\n\
        \t------------------------------------------\n \
        \t\tLaunching: LQR Tracker\n \
        \t------------------------------------------')
    
    x_size = x_ref.shape[0]
    u_size = u_ref.shape[0]
    TT = x_ref.shape[1]

    x_regulator = np.zeros((x_size, TT))
    u_regulator = np.zeros((u_size, TT))
    x_natural_evolution = np.zeros((x_size, TT))
    x_evolution_after_LQR = np.zeros((x_size, TT))

    x_regulator[:, 0]         = x_ref[:,0]
    x_evolution_after_LQR[:,0]= x_ref[:,0]

    u_regulator = u_ref

    delta_x = x_regulator - x_ref
    delta_u = u_regulator - u_ref

    Qt = np.zeros((x_size, x_size,TT-1))
    Rt = np.zeros((u_size, u_size,TT-1))

    K_Star = np.zeros((u_size, x_size, TT-1))

    A = np.zeros((x_size,x_size,TT-1))
    B = np.zeros((x_size,u_size,TT-1))

    # Calcolo le jacobiane di dinamica e costo
    for t in range(TT-1):
        A[:,:,t] = dyn.jacobian_x_new_wrt_x(x_ref[:,t], u_ref[:,t])
        B[:,:,t] = dyn.jacobian_x_new_wrt_u(x_ref[:,t])
        Qt[:,:,t] = cost.hessian1_J(t)           
        Rt[:,:,t] = cost.hessian2_J(t)              
    QT = cost.hessian_terminal_cost()

    K_Star = LQR_solver(A, B, Qt, Rt, QT)

    for t in range(TT-1):
        delta_u[:, t]  = K_Star[:,:,t] @ delta_x[:, t]
        delta_x[:, t+1]= A[:,:,t] @ delta_x[:,t] + B[:,:,t] @ delta_u[:, t]
    
    for t in range(TT-1):
        u_regulator[:,t] = u_ref[:,t] + delta_u[:,t]
        x_regulator[:,t] = x_ref[:,t] + delta_x[:,t]
        x_evolution_after_LQR[:,t+1] = dyn.dynamics(x_evolution_after_LQR[:,t], u_regulator[:,t])
    return u_regulator

def LQR_solver(A, B, Qt_Star, Rt_Star, QT_Star):
    x_size = A.shape[0]
    u_size = B.shape[1]
    TT = A.shape[2]+1
    
    delta_x = np.zeros((x_size,TT))

    P = np.zeros((x_size,x_size,TT))
    Pt = np.zeros((x_size,x_size))
    Ptt= np.zeros((x_size,x_size))
    
    K = np.zeros((u_size,x_size,TT-1))
    Kt= np.zeros((u_size,x_size))

    ######### Solve the Riccati Equation [S6C4]
    P[:,:,-1] = QT_Star

    for t in reversed(range(TT-1)):
        At  = A[:,:,t]
        Bt  = B[:,:,t]
        Qt  = Qt_Star[:,:,t]
        Rt  = Rt_Star[:,:,t]
        Ptt = P[:,:,t+1]

        temp = (Rt + Bt.T @ Ptt @ Bt)
        inv_temp = inv(temp)
        Kt =-inv_temp @ (Bt.T @ Ptt @ At)
        Pt = At.T @ Ptt @ At + At.T@ Ptt @ Bt @ Kt + Qt

        K[:,:,t] = Kt
        P[:,:,t] = Pt 
    return K

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
            anim(x_optimal[:,:,k].T)



def second_plot_set(k, sigma_star, delta_u, K_Star):
    if pm.Newton_Optcon_Plots and k % pm.Newton_Plot_every_k_iterations== 0:
            x_size = K_Star.shape[1]
            u_size = K_Star.shape[0]
            plt.figure()
            plt.title(f'Affine LQR solution at\nIteration {k}')
            for i in range(u_size):
                plt.plot(sigma_star[i,:, k], color = 'red', label = f'Sigma[{i}]')
                plt.plot(delta_u[i,:, k], color = 'purple', label = f'Delta_u[{i}]')
                for j in range(x_size):
                    plt.plot(K_Star[i, j, :, k], color = 'blue', label = f'K[{i} , {j}]')
            plt.grid()
            plt.legend()
            plt.show()
            

def plot_optimal_trajectory(x_reference, u_reference, x_gen, u_gen):
    
    total_time_steps = x_reference.shape[1]
    time = np.linspace(0, total_time_steps - 1, total_time_steps)  
    
    fig = plt.figure(figsize=(10, 10))
    
    names = {
        0: r'$\dot \theta_1$', 1: r'$\dot \theta_2$', 
        2: r'$\theta_1$', 3: r'$\theta_2$'
    }
    colors_ref = {0: 'm', 1: 'orange', 2: 'b', 3: 'g'}
    colors_opt = {0: 'purple', 1: 'chocolate', 2: 'navy', 3: 'lime'}
    
    for i in range(4):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.plot(time, x_reference[i, :], color=colors_ref[i], linestyle='--', label=f'{names[i]} (ref)', linewidth=2)
        ax.plot(time, x_gen[i, :], color=colors_opt[i], label=f'{names[i]} (opt)', linewidth=2)
        ax.set_title(names[i])
        ax.set_ylabel('[rad/s]' if i < 2 else '[rad]')
        ax.legend()
        ax.grid(True)
    
    ax = fig.add_subplot(3, 1, 3) 
    ax.plot(time, u_reference[0, :], color='r',  linestyle='--', label=r'$\tau_1$ (ref)', linewidth=2)
    ax.plot(time[:total_time_steps-1], u_gen[0,:total_time_steps-1], color='darkred', label=r'$\tau_1$ (opt)', linewidth=2)
    ax.set_title(r'$\tau_1$ Comparison')
    ax.set_xlabel('Time [steps]')
    ax.set_ylabel('[Nm]')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_optimal_intermediate_trajectory(x_reference, u_reference, x_gen, u_gen, k_max):
    
    total_time_steps = x_reference.shape[1]
    time = np.linspace(0, total_time_steps - 1, total_time_steps) 

    names = {
        0: r'$\dot \theta_1$', 1: r'$\dot \theta_2$', 
        2: r'$\theta_1$', 3: r'$\theta_2$', 4: r'$\tau_1$'
    }
    colors_ref = {0: 'm', 1: 'orange', 2: 'b', 3: 'g', 4: 'r'}
    colors_gen = {0: 'darkmagenta', 1: 'chocolate', 2: 'navy', 3: 'limegreen', 4: 'darkred'}

    selected_iterations = [1, 3, 7, 11]

    for i in range(4): 
        fig, axes = plt.subplots(len(selected_iterations), 1, figsize=(6, 10))

        for idx, k in enumerate(selected_iterations):
            if k > k_max:
                continue

            ax = axes[idx] if len(selected_iterations) > 1 else axes
            ax.plot(time, x_reference[i, :], color=colors_ref[i], linestyle='--', label=f'{names[i]} (ref)', linewidth=2)
            ax.plot(time, x_gen[i, :, k], color=colors_gen[i], label=f'{names[i]} (opt, k={k})', linewidth=2)

            ax.set_title(f'{names[i]} (Iteration {k})')
            ax.set_ylabel('[rad/s]' if i < 2 else '[rad]')
            ax.set_xlabel('Time [steps]')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    fig, axes = plt.subplots(len(selected_iterations), 1, figsize=(6, 10))

    for idx, k in enumerate(selected_iterations):
        if k > k_max:
            continue

        ax = axes[idx] if len(selected_iterations) > 1 else axes
        ax.plot(time, u_reference[0, :], color=colors_ref[4], linestyle='--', label=r'$\tau_1$ (ref)', linewidth=2)
        ax.plot(time[:total_time_steps - 1], u_gen[0, :total_time_steps - 1, k], color='darkred', label=fr'$\tau_1$ (opt, k={k})', linewidth=2)

        ax.set_title(r'$\tau_1$' f'(Iteration {k})')
        ax.set_ylabel('[Nm]')
        ax.set_xlabel('Time [steps]')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_norm_delta_u(delta_u_history):
    """
    Plot the norm of delta_u at each iteration k in semi-logarithmic scale.

    Parameters:
    delta_u_history (list): List of delta_u matrices for each iteration.
    """
    # Compute the Frobenius norm for each iteration's delta_u matrix
    norms = [np.linalg.norm(delta_u.flatten()) for delta_u in delta_u_history]
    iterations = np.arange(len(delta_u_history))

    plt.figure(figsize=(7, 4))
    # Points and line in mediumblue
    plt.semilogy(iterations, norms, '.', color='indigo', markersize=8)
    plt.semilogy(iterations, norms, '-', color='indigo', linewidth=2, alpha=0.8,
                 label=r'$\|\Delta u_k\|$')

    # Customize grid
    plt.grid(True, which="both", ls=":", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.4)
    
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'$\|\Delta u_k\|$', fontsize=12)
    
    # Improve legend with only line
    plt.legend(loc='upper right', fontsize=11, framealpha=1.0, 
              edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    plt.show()

def plot_norm_grad_J(grad_J_history):
    """
    Plot the norm of grad_J at each iteration k in semi-logarithmic scale.

    Parameters:
    grad_J_history (list or np.ndarray): List of grad_J vectors for each iteration.
    """
    norms = [np.linalg.norm(grad_J) for grad_J in grad_J_history]
    iterations = np.arange(len(grad_J_history))

    plt.figure(figsize=(7, 4))
    # Points and line in mediumblue
    plt.semilogy(iterations, norms, '.', color='indigo', markersize=8)
    plt.semilogy(iterations, norms, '-', color='indigo', linewidth=2, alpha=0.8,
                 label=r'$\|\nabla J(u^k)\|$')

    # Customize grid
    plt.grid(True, which="both", ls=":", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.4)
    
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'$\|\nabla J(u^k)\|$', fontsize=12)
    
    # Improve legend with only line
    plt.legend(loc='upper right', fontsize=11, framealpha=1.0,
              edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    plt.show()

def plot_cost_evolution(cost_history):
    """
    Plot the cost value at each iteration k in semi-logarithmic scale to show descent.

    Parameters:
    cost_history (list or np.ndarray): List of cost values for each iteration.
    """
    iterations = np.arange(len(cost_history))

    plt.figure(figsize=(7, 4))
    
    # Points and line in indigo
    plt.semilogy(iterations, cost_history, '.', color='indigo', markersize=8)
    plt.semilogy(iterations, cost_history, '-', color='indigo', linewidth=2, alpha=0.8,
                 label=r'$J(u^k)$')

    # Customize grid
    plt.grid(True, which="both", ls=":", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.4)
    
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'$J(u^k)$', fontsize=12)
    plt.title('Cost Evolution', fontsize=12)
    
    # Improve legend with only line
    plt.legend(loc='upper right', fontsize=11, framealpha=1.0,
              edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    plt.show()
