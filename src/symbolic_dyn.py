import sympy as sp

# Symbolic variables
M1, M2, L1, L2, R1, R2, I1, I2, G, F1, F2, dt = sp.symbols('M1 M2 L1 L2 R1 R2 I1 I2 G F1 F2 dt')
tau1, tau2, tau3, tau4 = sp.symbols('tau1 tau2 tau3 tau4')
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
u = sp.Matrix([[tau1], [tau2], [tau3], [tau4]])
u_reduced = sp.Matrix([[tau1]])

x_dot = A*x + B*u - M_inv_ext*(C_ext + G_ext) # x_new = f(x, u) 
x_dot = sp.simplify(x_dot)

x_new = x + x_dot*dt

x_new = x_new.subs({tau2: 0, tau3: 0, tau4: 0})

jacobian_x_new_wrt_x = x_new.jacobian(x)
jacobian_x_new_wrt_x = jacobian_x_new_wrt_x.subs({tau2: 0, tau3: 0, tau4: 0})

jacobian_x_new_wrt_u = x_new.jacobian(u_reduced)

hessians_x_new_wrt_x = [sp.hessian(f, x) for f in x_new]
hessians_x_dot_wrt_x = [hessian.subs({tau2: 0, tau3: 0, tau4: 0}) for hessian in hessians_x_new_wrt_x]
hessians_x_new_wrt_u = [sp.hessian(f, u_reduced) for f in x_new]
hessians_x_dot_wrt_u = [hessian.subs({tau2: 0, tau3: 0, tau4: 0}) for hessian in hessians_x_new_wrt_u]


mixed_hessians_x_new = []
for f in x_new:
   f = sp.Matrix([f])
   jacobian_wrt_x = f.jacobian(x)
   hessian_mixed = jacobian_wrt_x.jacobian(u_reduced)
   mixed_hessians_x_new.append(hessian_mixed)

# print("Matrix M:")
# sp.pprint(M)
# print("\n\n\n\n\n\nMatrix M_inv:")
# sp.pprint(M_inv)
# print("\n\n\n\n\n\nMatrix C:")
# sp.pprint(C)
# print("\n\n\n\n\n\nMatrix G:")
# sp.pprint(G)
# print("\n\n\n\n\n\nMatrix F:")
# sp.pprint(F)
# print("\n\n\n\n\n\nMatrix A:")
# sp.pprint(A)
# print("\n\n\n\n\n\nMatrix B:")
# sp.pprint(B)
# print("\n\n\n\n\n\nMatrix C_ext:")
# sp.pprint(C_ext)
# print("\n\n\n\n\n\nMatrix G_ext:")
# sp.pprint(G_ext)

############################################################################################################

print("\n\n\n\n\n\nMatrix x_new:")
sp.pprint(x_new)
for i in range(x_new.shape[0]):
    print(f"x_new[{i}] = {sp.ccode(x_new[i])}")

############################################################################################################

print("\n\n\n\n\n\nJacobian matrix of x_new with respect to x:")
subexprs, optimized_function = sp.cse(jacobian_x_new_wrt_x, symbols=sp.numbered_symbols(prefix='tmp'))
print("# Common Subexpressions:")
for var, expr in subexprs:
    print(f"{sp.ccode(var)} = {sp.ccode(expr)}")
    
function_matrix = optimized_function[0]

print("# Jacobian Elements:")
rows, cols = function_matrix.shape
for i in range(rows):
    for j in range(cols):
        print(f"dfx[{i},{j}] = {sp.ccode(function_matrix[i, j])}")
        
############################################################################################################
        
print("\n\n\n\n\n\nJacobian matrix of x_new with respect to u:")
subexprs, optimized_function = sp.cse(jacobian_x_new_wrt_u, symbols=sp.numbered_symbols(prefix='tmp'))
print("# Common Subexpressions:")
for var, expr in subexprs:
    print(f"{sp.ccode(var)} = {sp.ccode(expr)}")
    
function_matrix = optimized_function[0]

print("# Jacobian Elements:")
rows, cols = function_matrix.shape
for i in range(rows):
    for j in range(cols):
        print(f"dfu[{i},{j}] = {sp.ccode(function_matrix[i, j])}")
        
############################################################################################################

print("\n\n\n\n\n\nHessians of x_new with respect to x:")
subexprs, optimized_hessians = sp.cse(hessians_x_new_wrt_x, symbols=sp.numbered_symbols(prefix='tmp'))

print("# Common Subexpressions:")
for var, expr in subexprs:
    print(f"{sp.ccode(var)} = {sp.ccode(expr)}")

print("# Hessian Elements:")
hessian_index = 0
for hessian in optimized_hessians:
    print(f"# Hessian {hessian_index}:")
    rows, cols = hessian.shape
    for i in range(rows):
        for j in range(cols):
            print(f"Hxx[{hessian_index},{i},{j}] = {sp.ccode(hessian[i, j])}")
    hessian_index += 1
    
############################################################################################################

print("\n\n\n\n\n\nHessians of x_new with respect to u:")
subexprs, optimized_hessians = sp.cse(hessians_x_new_wrt_u, symbols=sp.numbered_symbols(prefix='tmp'))

print("# Common Subexpressions:")
for var, expr in subexprs:
    print(f"{sp.ccode(var)} = {sp.ccode(expr)}")
    
print("# Hessian Elements:")
hessian_index = 0
for hessian in optimized_hessians:
    print(f"# Hessian {hessian_index}:")
    rows, cols = hessian.shape
    for i in range(rows):
        for j in range(cols):
            print(f"Huu[{hessian_index},{i},{j}] = {sp.ccode(hessian[i, j])}")
    hessian_index += 1
    
############################################################################################################

print("\n\n\n\n\n\nMixed Hessians of x_new:")
subexprs, optimized_mixed_hessians = sp.cse(mixed_hessians_x_new, symbols=sp.numbered_symbols(prefix='tmp'))

print("# Common Subexpressions:")
for var, expr in subexprs:
    print(f"{sp.ccode(var)} = {sp.ccode(expr)}")
    
print("# Mixed Hessian Elements:")
hessian_index = 0
for hessian in optimized_mixed_hessians:
    print(f"# Mixed Hessian {hessian_index}:")
    rows, cols = hessian.shape
    for i in range(rows):
        for j in range(cols):
            print(f"Hxu[{hessian_index},{i},{j}] = {sp.ccode(hessian[i, j])}")
    hessian_index += 1