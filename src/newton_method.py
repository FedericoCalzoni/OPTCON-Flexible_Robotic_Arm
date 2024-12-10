import numpy as np
from dynamics import compute_gravity

def newton_method(z_0, jacobian_function, step_size=1e-5, iterations=1000):
    z = z_0
    tolerance = 1e-5
    for i in range(iterations):
        J_z = jacobian_function(*z.flatten())
        r_z = compute_gravity(*z.flatten())
        try:
            # TODO: verify if we need the transpose and if we
            # have the gravity upside down (change the sign)
            delta_z = - step_size * np.linalg.inv(J_z) @ r_z
            z = z + delta_z
            print(f"Iteration: {i}, z: {z}")
        except np.linalg.LinAlgError:
            print("Singular matrix at iteration: ", i)
            break
        if np.linalg.norm(delta_z) < tolerance:
            print(f"Converged after {i} iterations")
            break
        
    return z
