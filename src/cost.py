import numpy as np
from parameters import Qt, Rt, QT

def J_Function(x_trajectory, u_trajectory, x_reference, u_reference, type):
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
        J = J + stage_cost(x_trajectory[:, i], 
                          x_reference[:, i], 
                          u_trajectory[:, i], 
                          u_reference[:, i], type)
    J = J + terminal_cost(x_trajectory[:, T-1], x_trajectory[:, T-1], type)
    
    return J


def stage_cost(x_stage, x_reference, u_stage, u_reference, type):
    delta_x = x_stage - x_reference
    delta_u = u_stage - u_reference

    match type:
        case "Affine":
            qt = grad1_J(x_stage, x_reference)
            rt = grad2_J(u_stage, u_reference)
            J_t =  qt.T @ delta_x\
                +  rt.T @ delta_u\
                + (1/2) * delta_x.T @ Qt @delta_x \
                + (1/2) * delta_u.T @ Rt @delta_u

        case "Augmented":
            # Define Qt_Tilde
            qt = grad1_J(x_stage, x_reference)
            zeroblock = np.zeros((1,1))
            Qt_tilde = np.block([
                [zeroblock, qt.T],
                [qt, Qt]
                ])

            # Define delta_x_tilde
            oneblock = np.ones([1,1])
            delta_x_tilde = np.block([
                [oneblock], 
                [delta_x]
                ])

            # Define the decision variable vector
            decision_vector = np.block([
                [delta_x_tilde],
                [delta_u]
                ]) 

            # Define S_tilde
            rt = grad2_J(u_stage, u_reference)
            St = np.zeros([2, 4])
            St_tilde = np.block([
                [rt, St]
            ])

            # Define the block matrix mad of Q, S and R
            augmented_state_matrix = np.block([
                [Qt_tilde, St_tilde.T]
                [St_tilde, Rt]
            ])

            # Compute the stage cost 
            J_t = (1/2) * decision_vector.T @ augmented_state_matrix @ decision_vector 

        case "LQR":
            J_t = (1/2) * delta_x.T @ Qt @delta_x \
                + (1/2) * delta_u.T @ Rt @delta_u

    return J_t                                                                              
                                                                                
def terminal_cost(x_stage, x_reference,type):
    delta_x = x_stage - x_reference
    match type:
        case "Affine":
            qT = grad1_J(x_stage, x_reference)
            J_t =  qT.T @ delta_x\
                + (1/2) * delta_x.T @ QT @delta_x 
        case "Augmented":
            # Define Qt_Tilde
            qt = grad1_J(x_stage, x_reference)
            zeroblock = np.zeros((1,1))
            QT_tilde = np.block([
                [zeroblock, qt.T],
                [qt, Qt]
                ])

            # Define delta_x_tilde
            oneblock = np.ones([1,1])
            delta_x_tilde = np.block([
                [oneblock], 
                [delta_x]
                ])
            J_T = (1/2) * delta_x_tilde.T @ QT @ delta_x_tilde

        case "LQR":
            J_T = (1/2) * delta_x.T @ QT @ delta_x

    return J_T

def grad1_J(x_trajectory, x_reference):
    """
    Computes the gradient with respect to x of the cost function.
    
    Args:
        x_trajectory (numpy.ndarray): State trajectory vector of shape (4,)
        x_reference (numpy.ndarray): Reference state trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (x_trajectory - x_reference)
    """
    return Qt @ (x_trajectory - x_reference)

def grad2_J(u_trajectory, u_reference):
    """
    Computes the gradient with respect to u of the cost function.
    
    Args:
        u_trajectory (numpy.ndarray): Input trajectory vector of shape (4,)
        u_reference (numpy.ndarray): Reference Input trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (u_trajectory - u_reference)
    """
    return Rt @ (u_trajectory - u_reference)

def grad_terminal_cost(xT, xT_reference):
    """
    Computes the gradient with respect to x of the terminal cost function.
    
    Args:
        xT (numpy.ndarray): State trajectory vector of shape (4,)
        xT_reference (numpy.ndarray): Reference state trajectory vector of shape (4,)

    Returns:
        float: The gradient of the cost function computed in (xT - xT_reference)
    """
    return QT @ (xT - xT_reference)

def hessian1_J():
    """
    Computes the Hessian of the cost function with respect to x.
    
    Args:
        Qt (numpy.ndarray): Weight matrix of shape (4, 4).

    Returns:
        numpy.ndarray: The Hessian of the cost function, which is equal to Qt.
    """
    return Qt

def hessian2_J():
    """
    Computes the Hessian of the cost function with respect to u.
    
    Args:
        Rt (numpy.ndarray): Weight matrix of shape (4, 4).

    Returns:
        numpy.ndarray: The Hessian of the cost function, which is equal to Rt.
    """
    return Rt

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

def hessian_terminal_cost():
    """
    Computes the Hessian of the terminal cost function with respect to x.
    
    Args:
        QT (numpy.ndarray): Weight matrix of shape (4, 4).
 
    Returns:
        numpy.ndarray: The Hessian of the terminal cost function, which is equal to QT.
    """
    return QT
 