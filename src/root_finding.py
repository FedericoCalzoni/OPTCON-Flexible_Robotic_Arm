import numpy as np
from newton_method import newton_method

# Dynamics parameters
M1 = 2
M2 = 2
L1 = 1.5
L2 = 1.5
R1 = 0.75
R2 = 0.75
I1 = 1.5
I2 = 1.5
G = 9.81
F1 = 0.1
F2 = 0.1

# Jacobian of G
def jacobian_G(theta1, theta2):
    JG_11 = G*M2*R2*np.cos(theta1 + theta2) + G*(L1*M2 + M1*R1)*np.cos(theta1)
    JG_12 = G*M2*R2*np.cos(theta1 + theta2)
    JG_21 = JG_12
    JG_22 = JG_12
    return np.array([[JG_11 , JG_12], [JG_21 , JG_22]])

z_0 = np.array([[np.pi/3], [np.pi/3]])

eq = newton_method(z_0, jacobian_G)

print("eq: ", eq)

