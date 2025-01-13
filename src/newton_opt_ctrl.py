import numpy as np
import dynamics as dyn
import parameters as pm
import matplotlib.pyplot as plt
from numpy.linalg import inv
import armijo as armijo
from visualizer import animate_double_pendulum as anim

def newton_for_optcon(x_reference, u_reference, guess="step", task=1):
    
    if task == 1:
        import costTask1 as cost
    elif task == 2:
        import costTask2 as cost

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

    if  guess == "step":
        for t in range(TT):
            x_optimal[:,t, 0] = x_reference[:,0]
            u_optimal[:,t, 0] = u_reference[:,0]
    elif guess == "smooth":
        x_optimal[:,:, 0] = x_reference
        u_optimal[:,:, 0] = Initial_LQR(x_reference, u_reference)
    else:
        raise ValueError("Invalid guess parameter: must be 'first equilibria' or 'reference'")

    # Apply newton method to compute the optimal trajectory
    GradJ_u_history = []  # Store GradJ_u history
    delta_u_history = []  # Store delta_u history
    
    for k in range(max_iterations):
        l[k] = cost.J_Function(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, "LQR")
        
        # Gradient norm stopping criteria
        if k <= 1:
            print(f"\nIteration: {k} \tCost: {l[k]}")
        else: 
            norm_delta_u =  np.linalg.norm(delta_u[:,:,k-1])
            print(f"\nIteration: {k} \tCost: {l[k]}\tCost reduction: {l[k] - l[k-1]}\tDelta_u Norm: {norm_delta_u}")
            if norm_delta_u < 1e-3:
                break
    
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
        
        GradJ_u_history.append(GradJ_u.copy())

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
        
        delta_u_history.append(delta_u[:,:,k].copy())
        
        # Compute step size
        if k == 0: #and guess == "smooth":
            gamma = 0.1
        else:
            gamma = armijo.armijo(x_optimal[:,:,k], x_reference, u_optimal[:,:,k], u_reference, delta_u[:,:,k], GradJ_u, l[k], K_Star[:,:,:,k], sigma_star[:,:,k], k, task, step_size_0=1)

        for t in range(TT-1):
            u_optimal[:,t, k+1] = u_optimal[:,t, k] + K_Star[:,:,t, k] @ (x_optimal[:,t, k+1] - x_optimal[:,t,k]) + gamma * sigma_star[:,t, k]
            x_optimal[:,t+1, k+1] = dyn.dynamics(x_optimal[:,t,k+1], u_optimal[:,t,k+1])

    print(f'Algorithm Ended at {k}th iteration')
    newton_finished = True
    plot_optimal_intermediate_trajectory(x_reference, u_reference, x_optimal, u_optimal, k)
    return x_optimal[:,:,k], u_optimal[:,:,k], GradJ_u_history, delta_u_history, l[:k+1]

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
    
    import costTask3 as cost
    
    x_size = x_ref.shape[0]
    u_size = u_ref.shape[0]
    TT = x_ref.shape[1]

    x_regulator = np.zeros((x_size, TT))
    u_regulator = np.zeros((u_size, TT))
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

    plotsteps = int((k_max - 2)/2)
    selected_iterations = [1, plotsteps, 2*plotsteps, k_max]

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
    iterations = np.arange(len(delta_u_history)-1)

    plt.figure(figsize=(7, 4))
    # Points and line in mediumblue
    plt.semilogy(iterations, norms[1:], '.', color='indigo', markersize=8)
    plt.semilogy(iterations, norms[1:], '-', color='indigo', linewidth=2, alpha=0.8,
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
    valid_indices = [i for i, grad_J in enumerate(grad_J_history) if np.linalg.norm(grad_J) != 0]
    valid_grad_J_history = [grad_J_history[i] for i in valid_indices]

    # Compute norms of valid gradients
    norms = [np.linalg.norm(grad_J) for grad_J in valid_grad_J_history]
    iterations = np.array(valid_indices)

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
    iterations = np.arange(len(cost_history)-1)

    plt.figure(figsize=(7, 4))
    
    # Points and line in indigo
    plt.semilogy(iterations, cost_history[1:], '.', color='indigo', markersize=8)
    plt.semilogy(iterations, cost_history[1:], '-', color='indigo', linewidth=2, alpha=0.8,
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
    
# def plot_error_evolution(x_gen, u_gen, x_reference, u_reference):
#     delta_x = np.zeros((4,pm.TT))
#     for i in range(x_reference.shape[0]):
#         delta_x[i, :] =  x_gen[i, :] - x_reference[i, :] 
#         plt.plot(delta_x[i, :], label = f'delta x{i}')
#     delta_u = u_gen - u_reference
#     plt.plot(delta_u[0, :], label = 'delta u')
#     plt.title('Error Evolution')
#     plt.legend()
#     plt.grid()
#     plt.show()
