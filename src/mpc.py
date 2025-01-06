import numpy as np
import cvxpy as cp
from dynamics import dynamics, jacobian_x_new_wrt_x, jacobian_x_new_wrt_u
import cost
from parameters import T_pred, umax, umin

def solver_linear_mpc(A, B, Q, R, x_t, t_start): 

    x_size, u_size, T = B.shape

    # Define decision variables
    x_mpc = cp.Variable((x_size, T_pred))
    u_mpc = cp.Variable((u_size, T_pred))

    # Define cost and constraints
    cost = 0
    constraints = []


    for tau in range(T_pred - 1):
        if t_start + tau >= T:
            A_tau = A[:,:,-1]
            B_tau = B[:,:,-1]
            Q_tau = Q[:,:,-1]
            R_tau = R[:,:,-1]
        else:
            A_tau = A[:,:,t_start+tau]
            B_tau = B[:,:,t_start+tau]
            Q_tau = Q[:,:,t_start+tau]
            R_tau = R[:,:,t_start+tau]
                  
        cost += cp.quad_form(x_mpc[:, tau] - x_mpc[:, tau], Q_tau) + cp.quad_form(u_mpc[:, tau] - u_mpc[:, tau], R_tau)
        constraints += [
            x_mpc[:, tau + 1] == A_tau @ x_mpc[:, tau] + B_tau @ u_mpc[:, tau],
            u_mpc[:, tau] <= umax,
            u_mpc[:, tau] >= umin,
        ]

    # sums problem objectives and concatenates constraints
    cost += cp.quad_form(x_mpc[:, T_pred - 1] - x_mpc[:, T_pred - 1], Q[:,:,-1])
    constraints += [x_mpc[:, 0] == x_t]

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return x_mpc.value, u_mpc.value

def compute_mpc(x_gen, u_gen):

    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    T = x_gen.shape[1]
    A = np.zeros((x_size, x_size, T))
    B = np.zeros((x_size, u_size, T))

    Q = np.zeros((x_size, x_size, T))
    R = np.zeros((u_size, u_size, T))
    
    x_real_mpc = np.zeros((x_size, T))
    u_real_mpc = np.zeros((u_size, T))

    x_mpc = np.zeros((x_size, T_pred, T))
    u_mpc = np.zeros((u_size, T_pred, T))
    x_real_mpc[:,0] = x_gen[:,0]

    for t in range(T-1):
        A[:,:,t] = jacobian_x_new_wrt_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = jacobian_x_new_wrt_u(x_gen[:,t])

        Q[:,:,t] = cost.hessian1_J()           
        R[:,:,t] = cost.hessian2_J()              
    Q[:,:,-1] = cost.hessian_terminal_cost()

    for t in range(T-1):
        x_t_mpc = x_real_mpc[:,t]
        x_mpc[:,:,t], u_mpc[:,:,t] = solver_linear_mpc(A, B, Q, R, x_t_mpc, t)
        u_real_mpc[:,t] = u_mpc[:,0,t]
    
        x_real_mpc[:,t+1] = dynamics(x_real_mpc[:,t], u_real_mpc[:,t])

    return x_real_mpc, u_real_mpc    
        