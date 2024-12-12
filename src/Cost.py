import numpy as np
from numpy import transpose as Tr
from parameters import Qt, Rt, QT

def J_Function(x_trajectory, u_trajectory, x_reference, u_reference):
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

    for i in range (T - 2):
        J = J + StageCost(x_trajectory[:, i], x_reference[:, i], u_trajectory[:, i], u_reference[:, i])

    J = J + TerminalCost(x_trajectory[:, T-1], x_trajectory[:, T-1])
    return J

def StageCost(x_stage, x_reference, u_stage, u_reference):
    
    Jt = (1/2)*(Tr(x_stage - x_reference))@ Qt @(x_stage - x_reference) + (1/2)*(Tr(u_stage - u_reference) @ Rt @ (u_stage - u_reference))
    return Jt                                                                              
                                                                                
def TerminalCost(xT, xT_reference):
    JT = (1/2)*(Tr(xT - xT_reference))@ QT @(xT - xT_reference)
    return JT

def Grad1J(x_trajectory, x_reference):
    """
    Computes the gradient with respect to x of the cost function.
    
    Args:
        x_trajectory (numpy.ndarray): State trajectory vecto of shape (4,)
        x_reference (numpy.ndarray): Reference state trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (x_trajectory - x_reference)
    """
    return Qt @ (x_trajectory - x_reference)

def Grad2J(u_trajectory, u_reference):
    """
    Computes the gradient with respect to u of the cost function.
    
    Args:
        u_trajectory (numpy.ndarray): Input trajectory vecto of shape (4,)
        u_reference (numpy.ndarray): Reference Input trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (u_trajectory - u_reference)
    """
    return Rt @ (u_trajectory - u_reference)
