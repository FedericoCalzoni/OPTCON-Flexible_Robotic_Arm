import numpy as np
import cvxpy as cp
from dynamics import dynamics, jacobian_x_new_wrt_x, jacobian_x_new_wrt_u

def solver_linear_mpc(A, B, Q, R, Qf, x_t, x_ref, u_ref, umax=50, umin=-50, T_pred=5): 

    ns, ni = B.shape

    # Define decision variables
    x_mpc = cp.Variable((ns, T_pred))
    u_mpc = cp.Variable((ni, T_pred))

    # Define cost and constraints
    cost = 0
    constraints = []

    for t in range(T_pred - 1):
        cost += cp.quad_form(x_mpc[:, t] - x_ref[:, t], Q) + cp.quad_form(u_mpc[:, t] - u_ref[:, t], R)
        constraints += [
            x_mpc[:, t + 1] == A @ x_mpc[:, t] + B @ u_mpc[:, t],
            u_mpc[:, t] <= umax,
            u_mpc[:, t] >= umin,
        ]

    # sums problem objectives and concatenates constraints
    cost += cp.quad_form(x_mpc[:, T_pred - 1] - x_ref[:, T_pred - 1], Qf)
    constraints += [x_mpc[:, 0] == x_t]

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return u_mpc[:, 0].value, x_mpc.value, u_mpc.value