import sympy as sp

# Symbolic variables
M1, M2, L1, L2, R1, R2, I1, I2, G, F1, F2 = sp.symbols('M1 M2 L1 L2 R1 R2 I1 I2 G F1 F2')
tau1, tau2, tau3, tau4= sp.symbols('tau1 tau2 tau3 tau4')
theta1, theta2, dtheta1, dtheta2 = sp.symbols('theta1 theta2 dtheta1 dtheta2')

def compute_inertia_matrix(theta2):
    """Compute the inertia matrix M."""
    cos_theta2 = sp.cos(theta2)
    m11 = I1 + I2 + M1 * R1**2 + M2 * (L1**2 + R2**2) + 2 * M2 * L1 * R2 * cos_theta2
    m12 = I2 + M2 * R2**2 + M2 * L1 * R2 * cos_theta2
    m21 = m12
    m22 = I2 + M2 * R2**2
    return sp.Matrix([[m11, m12], [m21, m22]])

def compute_coriolis(theta2, dtheta1, dtheta2):
    """Compute the Coriolis matrix C."""
    sin_theta2 = sp.sin(theta2)
    c1 = -M2 * L1 * R2 * dtheta2 * sin_theta2 * (dtheta2 + 2 * dtheta1)
    c2 = M2 * L1 * R2 * sin_theta2 * dtheta1**2
    return sp.Matrix([[c1], [c2]])

def compute_gravity(theta1, theta2):
    """Compute the gravity forces matrix G."""
    sin_theta1 = sp.sin(theta1)
    sin_theta1_theta2 = sp.sin(theta1 + theta2)
    g1 = G * (M1 * R1 + M2 * L1) * sin_theta1 + G * M2 * R2 * sin_theta1_theta2
    g2 = G * M2 * R2 * sin_theta1_theta2
    return sp.Matrix([[g1], [g2]])

M = compute_inertia_matrix(theta2)
C = compute_coriolis(theta2, dtheta1, dtheta2)
G = compute_gravity(theta1, theta2)

F = sp.Matrix([[F1, 0],
              [ 0 ,F2]])

M_inv = M.inv()

Z_2x2 = sp.zeros(2,2)

A = sp.BlockMatrix([[-M_inv @ F, Z_2x2],
                    [sp.eye(2), Z_2x2]])

A = A.as_explicit()

M_inv_ext = sp.BlockMatrix([[M_inv, Z_2x2],
                            [Z_2x2, Z_2x2]])

M_inv_ext = M_inv_ext.as_explicit()

B = M_inv_ext

C_ext = sp.BlockMatrix([[C],
                        [sp.zeros(2, 1)]])

C_ext = C_ext.as_explicit()

G_ext = sp.BlockMatrix([[G],
                        [sp.zeros(2, 1)]])

G_ext = G_ext.as_explicit()

x = sp.Matrix([[dtheta1], [dtheta2], [theta1], [theta2]])
u = sp.Matrix([[tau1],[tau2],[tau3],[tau4]])

x_dot = A*x + B*u - M_inv_ext*(C_ext + G_ext) # x_dot = f(x, u) 
x_dot = sp.simplify(x_dot)

jacobian_G = G.jacobian(x)
jacobian_x_dot_wrt_x = x_dot.jacobian(x)

jacobian_x_dot_wrt_u = x_dot.jacobian(u)



print("Matrix M:")
sp.pprint(M)
print("\n\n\n\n\n\nMatrix M_inv:")
sp.pprint(M_inv)
print("\n\n\n\n\n\nMatrix C:")
sp.pprint(C)
print("\n\n\n\n\n\nMatrix G:")
sp.pprint(G)
print("\n\n\n\n\n\nMatrix F:")
sp.pprint(F)
print("\n\n\n\n\n\nMatrix A:")
sp.pprint(A)
print("\n\n\n\n\n\nMatrix B:")
sp.pprint(B)
print("\n\n\n\n\n\nMatrix C_ext:")
sp.pprint(C_ext)
print("\n\n\n\n\n\nMatrix G_ext:")
sp.pprint(G_ext)
print("\n\n\n\n\n\nMatrix x_dot:")
sp.pprint(x_dot)
# print("\n\n\nJacobian matrix of x_dot with respect to x:")
# for i in range(jacobian_x_dot.shape[0]):
#     for j in range(jacobian_x_dot.shape[1]):
#         print(f"Element ({i+1}, {j+1}):")
#         sp.pprint(jacobian_x_dot[i, j])
#         print("")


print("\nGradient matrix of G with respect to x:")
for i in range(jacobian_G.shape[0]):
    for j in range(jacobian_G.shape[1]):
        print(f"Gradient element ({i+1}, {j+1}) = {jacobian_G[i, j]}")

print("\n\n\n\n\n\nJacobian matrix of x_dot with respect to x:")
for i in range(jacobian_x_dot_wrt_x.shape[0]):
    for j in range(jacobian_x_dot_wrt_x.shape[1]):
        print(f"Jacobian element ({i+1}, {j+1}) = {jacobian_x_dot_wrt_x[i, j]}")
        
print("\n\n\n\n\n\nJacobian matrix of x_dot with respect to u:")
for i in range(jacobian_x_dot_wrt_u.shape[0]):
    for j in range(jacobian_x_dot_wrt_u.shape[1]):
        print(f"Jacobian element ({i+1}, {j+1}) = {jacobian_x_dot_wrt_u[i, j]}")
        

