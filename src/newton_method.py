import numpy as np
from dynamics import compute_gravity
from dynamics import jacobian_gravity as jacobian_function

def newton_method(initial_guess, step_size=1e-2, iterations=1000):
    """
    Applies Newton's method to find the root of a function.

    Args:
        initial_guess (array-like): Initial guess for the variables.
        jacobian_function (callable): Function to compute the Jacobian matrix.
        step_size (float, optional): Step size for the update. Defaults to 1e-2.
        iterations (int, optional): Maximum number of iterations. Defaults to 1000.

    Returns:
        array-like: The estimated root of the function.

    Raises:
        np.linalg.LinAlgError: If the Jacobian matrix is singular.
    """
    z = initial_guess
    tolerance = 1e-5
    for i in range(iterations):
        z_0 = z[0].item()
        z_1 = z[1].item()
        z_2 = z[2].item()
        J_z = jacobian_function(z_0, z_1)
        r_z = compute_gravity(z_0, z_1)- [[z_2], [0]]
        # armijo could be used to increase the convergence speed
        try:
            delta_z = - step_size * np.linalg.pinv(J_z) @ r_z
            z = z + delta_z
        except np.linalg.LinAlgError:
            print("Singular matrix at iteration: ", i)
            break
        if np.linalg.norm(delta_z) < tolerance:
            print(f"Converged after {i} iterations")
            break
    
    x_eq = np.array([0, 0, z[0].item(), z[1].item()])
    u_eq = np.array([z[2].item()])

    return x_eq, u_eq
