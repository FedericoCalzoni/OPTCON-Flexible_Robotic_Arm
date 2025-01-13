import numpy as np
import dynamics as dyn
import parameters as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import inv
import armijo as armijo
import data_manager as dm
from visualizer import animate_double_pendulum as anim


import numpy as np

def J_Function(x_trajectory, u_trajectory, x_reference, u_reference, type, Q_tuned, R_tuned):
    """
    Computes the total cost of a trajectory for the current iteration.
    The cost is calculated using stage costs for each step of the trajectory and a terminal cost at the final step.

    Args:
        x_trajectory (numpy.ndarray): State trajectory matrix of shape (n, T),
            where `n` is the dimension of the state vector and `T` is the number of time steps.
        u_trajectory (numpy.ndarray): Input trajectory matrix of shape (m, T-1),
            where `m` is the dimension of the control vector.
        x_reference (numpy.ndarray): Reference state trajectory matrix of shape (n, T),
            representing the desired state trajectory.
        u_reference (numpy.ndarray): Reference input trajectory matrix of shape (m, T-1),
            representing the desired control inputs.

    Returns:
        float: The total cost of the trajectory, calculated as the sum of stage costs and the terminal cost.

    Note:
        - Matrices Qt, QT and Rt are defined in parameters.py.
    """
    J = 0
    T = x_trajectory.shape[1]
    for t in range (T - 2):
        J = J + stage_cost(x_trajectory[:, t], 
                          x_reference[:, t], 
                          u_trajectory[:, t], 
                          u_reference[:, t], type,t, Q_tuned, R_tuned)
    J = J + terminal_cost(x_trajectory[:, T-1], x_trajectory[:, T-1], type, Q_tuned)
    
    return J

def stage_cost(x_stage, x_reference, u_stage, u_reference, type, t, Q_tuned, R_tuned):
    delta_x = x_stage - x_reference
    delta_u = u_stage - u_reference

    match type:
        case "LQR":
            J_t = (1/2) * delta_x.T @ Q_tuned[:,:,t] @delta_x \
                + (1/2) * delta_u.T @ R_tuned[:,:,t] @delta_u

    return J_t                                                                              
                                                                                
def terminal_cost(x_stage, x_reference,type, Q_tuned):
    delta_x = x_stage - x_reference
    match type:
        case "LQR":
            J_T = (1/2) * delta_x.T @ Q_tuned[:,:,-1] @ delta_x

    return J_T


def grad1_J(x_trajectory, x_reference, t, Q_tuned):
    """
    Computes the gradient with respect to x of the cost function.
    
    Args:
        x_trajectory (numpy.ndarray): State trajectory vector of shape (4,)
        x_reference (numpy.ndarray): Reference state trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (x_trajectory - x_reference)
    """
    return Q_tuned[:,:,t] @ (x_trajectory - x_reference)

def grad2_J(u_trajectory, u_reference,t, R_tuned):
    """
    Computes the gradient with respect to u of the cost function.
    
    Args:
        u_trajectory (numpy.ndarray): Input trajectory vector of shape (4,)
        u_reference (numpy.ndarray): Reference Input trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (u_trajectory - u_reference)
    """
    return R_tuned[:,:,t] @ (u_trajectory - u_reference)

def grad_terminal_cost(xT, xT_reference, Q_tuned):
    """
    Computes the gradient with respect to x of the terminal cost function.
    
    Args:
        xT (numpy.ndarray): State trajectory vector of shape (4,)
        xT_reference (numpy.ndarray): Reference state trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (xT - xT_reference)
    """
    return Q_tuned[:,:,-1] @ (xT - xT_reference)

def hessian1_J(t, Q_tuned):
    """
    Computes the Hessian of the cost function with respect to x.
    
    Args:
        Qt (numpy.ndarray): Weight matrix of shape (4, 4).

    Returns:
        numpy.ndarray: The Hessian of the cost function, which is equal to Qt.
    """
    return Q_tuned[:,:,t]

def hessian2_J(t, R_tuned):
    """
    Computes the Hessian of the cost function with respect to u.
    
    Args:
        Rt (numpy.ndarray): Weight matrix of shape (4, 4).

    Returns:
        numpy.ndarray: The Hessian of the cost function, which is equal to Rt.
    """
    return R_tuned[:,:,t]

def hessian_12_J(x_trajectory, u_trajectory):
    """
    Computes the mixed second derivative of the cost function (derivative of grad1 with respect to u).
    
    Args:
        x_trajectory (numpy.ndarray): State trajectory vector of shape (4,)
        u_trajectory (numpy.ndarray): Input trajectory vector of shape (4,)

    Returns:
        numpy.ndarray: The mixed Hessian, which is a zero matrix.
    """
    return np.zeros((u_trajectory.shape[0], x_trajectory.shape[0]))

def hessian_terminal_cost(Q_tuned):
    """
    Computes the Hessian of the terminal cost function with respect to x.
    
    Args:
        QT (numpy.ndarray): Weight matrix of shape (4, 4).
 
    Returns:
        numpy.ndarray: The Hessian of the terminal cost function, which is equal to QT.
    """
    return Q_tuned[:,:,-1]
 


def tune_cost_matrices(x_reference, u_reference, num_iterations=10000, perturbation_scale=10):
    """
    Tunes the Qt_temp and Rt_temp matrices based on the objective function evaluated with newton_OC.
    
    Parameters:
        Qt_temp (np.array): Initial Qt matrix for both phases
        Rt_temp (np.array): Initial Rt matrix for both phases
        cost_matrices_computation (function): Function to compute Qt, Rt
        newton_OC (function): Objective function to evaluate cost
        TT (int): Time horizon or parameter required for cost_matrices_computation
        divisions (int): Number of divisions or other parameter for cost_matrices_computation
        transition_width (float): Transition width or other parameter for cost_matrices_computation
        num_iterations (int): Number of iterations to run the optimization
        perturbation_scale (float): Magnitude of random perturbations for slow tuning
        improvement_threshold (float): Threshold for considering improvements in cost function
        
    Returns:
        Best tuned Qt_temp and Rt_temp matrices.
    """
    Qt_temp = np.zeros((4, 4, 2))
    Rt_temp = np.zeros((1, 1, 2))
    # Qt_temp[:, :, 0] = np.diag([59565.50815028, 129466.83174896, 5000231.63033068, 299772.1354184])   # Constant phase
    # Rt_temp[:, :, 0] = np.diag([61.93269819])                                 # Constant phase
    # Qt_temp[:, :, 1] = np.diag([62.14345657, 28.26521521, 45.20548933, 25.94358565])                  # Transition phase
    # Rt_temp[:, :, 1] = np.diag([13.5312041]) 
    Qt_temp[:, :, 0] = np.diag([1.00001241, 99.9331399, 490.9935389, 1.00096131]) *1e8  # Constant phase
    Rt_temp[:, :, 0] = np.diag([265.5499998])                                 # Constant phase
    Qt_temp[:, :, 1] = np.diag([1.02688787, 0.00000001, 3.33155371, 4.92487656]) * 1                  # Transition phase
    Rt_temp[:, :, 1] = np.diag([299.40677386])
    TT = int((pm.t_f - pm.t_i)/pm.dt)
    transition_width = 3000
    divisions = 5

    # Initial cost calculation
    Qt, Rt = pm.cost_matrices_computation(Qt_temp, Rt_temp, TT, divisions, transition_width)
    current_cost = newton_for_optcon(x_reference, u_reference, Qt, Rt)
    print(f"TUNING COST: {current_cost}")
    
    for iteration in range(num_iterations):
        # Copy the matrices for perturbation
        Qt_temp_new = np.copy(Qt_temp)
        Rt_temp_new = np.copy(Rt_temp)
        
        # Add perturbations to diagonal elements only
        Qt_temp_new[:, :, 0] += np.diag(1000*perturbation_scale * np.random.randn(Qt_temp.shape[1]))
        Qt_temp_new[:, :, 1] += np.diag(0.1*perturbation_scale * np.random.randn(Qt_temp.shape[1]))
        Rt_temp_new[:, :, 0] += np.diag(perturbation_scale * np.random.randn(Rt_temp.shape[1]))
        Rt_temp_new[:, :, 1] += np.diag(0.1*perturbation_scale * np.random.randn(Rt_temp.shape[1]))
        
        Qt_temp_new[:, :, 0] = np.diag(np.maximum(np.diag(Qt_temp_new[:, :, 0]), 1e-8))
        Qt_temp_new[:, :, 1] = np.diag(np.maximum(np.diag(Qt_temp_new[:, :, 1]), 1e-8))
        Rt_temp_new[:, :, 0] = np.diag(np.maximum(np.diag(Rt_temp_new[:, :, 0]), 1e-8))
        Rt_temp_new[:, :, 1] = np.diag(np.maximum(np.diag(Rt_temp_new[:, :, 1]), 1e-8))

        
        # Recompute Qt and Rt with the new matrices
        Qt_new, Rt_new = pm.cost_matrices_computation(Qt_temp_new, Rt_temp_new, TT, divisions, transition_width)
        
        # Calculate the new cost
        new_cost = newton_for_optcon(x_reference, u_reference, Qt_new, Rt_new)
        print(f"NEW TUNING COST: {new_cost}")
        
        # Check if the cost has improved
        if new_cost > current_cost:
            Qt_temp = Qt_temp_new
            Rt_temp = Rt_temp_new
            current_cost = new_cost
            print(f"Iteration {iteration}: Cost improved to {new_cost}")
        else:
            print(f"Iteration {iteration}: No improvement, cost remains {current_cost}")
        
        print(f"Qt_temp: [{Qt_temp[0,0,0]}, {Qt_temp[1,1,0]}, {Qt_temp[2,2,0]}, {Qt_temp[3,3,0]}]")
        print(f"Rt_temp: [{Rt_temp[0,0,0]}]")
        print(f"Qt_temp: [{Qt_temp[0,0,1]}, {Qt_temp[1,1,1]}, {Qt_temp[2,2,1]}, {Qt_temp[3,3,1]}]")
        print(f"Rt_temp: [{Rt_temp[0,0,1]}]")
        
        # Optionally, we could print the matrices at intervals to observe progress
    
    return Qt_temp, Rt_temp


def newton_for_optcon(x_reference, u_reference, Q_tuned, R_tuned):
    print('\n\n\
    \t--------------------------------------------\n \
    \tLaunching: Newton Method for Optimal Control\n \
    \t--------------------------------------------')
    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]
    TT = x_reference.shape[1]
    max_iterations = 4
    
    l = np.zeros((max_iterations)) # Cost function
    x_initial_guess = x_reference[:,0]
    u_initial_guess = u_reference[:,0]

    x_optimal = np.zeros((x_size, TT, max_iterations+1))
    u_optimal = np.zeros((u_size, TT, max_iterations+1))

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
    relative_cost_sum = 0

    # Initialize the first instance of the optimal trajectory
    for t in range(TT):
        x_optimal[:,t, 0] = x_initial_guess
        u_optimal[:,t, 0] = u_initial_guess

    # Apply newton method to compute the optimal trajectory
    for k in range(max_iterations):
        
        l[k] = J_Function(x_optimal[:,:,k], u_optimal[:,:,k], x_reference, u_reference, "LQR", Q_tuned, R_tuned)

        if k == 0:
            print(f"\nIteration: {k} \tCost: {l[k]}")

        else:
            # Check if the terminal condition is satisfied
            relative_cost_reduction = np.abs(l[k] - l[k-1])/l[k-1]
            if relative_cost_reduction > 1:
                relative_cost_sum += -100000
            else:
                relative_cost_sum += relative_cost_reduction
            print(f"\nIteration: {k} \tCost: {l[k]} \tCost reduction: {l[k] - l[k-1]} \tRelative cost reduction: {relative_cost_reduction}")
            if relative_cost_reduction < 1e-10:
                break
        
        # Initialization of x0 for the next iteration
        x_optimal[:,0, k+1] = x_initial_guess
        
        # Compute the Jacobians of the dynamics and the cost
        for t in range(TT-1):
            A[:,:,t] = dyn.jacobian_x_new_wrt_x(x_optimal[:,t,k], u_optimal[:,t,k])
            B[:,:,t] = dyn.jacobian_x_new_wrt_u(x_optimal[:,t,k])
            qt[:,t] = Q_tuned[:,:,t] @ (x_optimal[:,t,k] - x_reference[:,t])
            rt[:,t] = R_tuned[:,:,t] @ (u_optimal[:,t,k] - u_reference[:,t])
        qT = Q_tuned[:,:,-1] @ (x_optimal[:,-1,k] - x_reference[:,-1])
        
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
            Qt_Star[:,:,t] = Q_tuned[:,:,t]        
            Rt_Star[:,:,t] = R_tuned[:,:,t]            
            St_Star[:,:,t] = hessian_12_J(x_optimal[:,t,k], u_optimal[:,t,k])  
        QT_Star = Q_tuned[:,:,-1]


        ########## Compute the optimal control input [S18C9]
        # To compute the descent direction, the affine LQR must be solved
        K_Star[:,:,:,k], sigma_star[:,:,k], delta_u[:,:,k] =  \
            Affine_LQR_solver(x_optimal[:,:,k], x_reference, A, B, \
                              Qt_Star, Rt_Star, St_Star, QT_Star, qt, rt, qT)
                    
        # Compute the proper stepsize
        #if k == 5:
            #breakpoint()
        gamma = armijo.armijo(x_optimal[:,:,k], x_reference, u_optimal[:,:,k], u_reference, delta_u[:,:,k], GradJ_u, l[k], K_Star[:,:,:,k], sigma_star[:,:,k], k, step_size_0=1)

        # gamma = 0.1

        for t in range(TT-1): 
            u_optimal[:,t, k+1] = u_optimal[:,t, k] + K_Star[:,:,t, k] @ (x_optimal[:,t, k+1] - x_optimal[:,t,k]) + gamma * sigma_star[:,t, k]
            u_optimal[:, t] = np.clip(u_optimal[:, t], -100, 100)
            x_optimal[:,t+1, k+1] = dyn.dynamics(x_optimal[:,t,k+1], u_optimal[:,t,k+1])

    print(f'Ho finito alla {k}^ iterazione')
    newton_finished = True
    return relative_cost_sum

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

x_reference, u_reference = dm.load_optimal_trajectory(version = '99')
Qt_temp, Rt_temp = tune_cost_matrices(x_reference, u_reference)


