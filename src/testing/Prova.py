from sympy import symbols, Matrix, sin, cos

# Definizione delle variabili
θ1, θ2, θ1_dot, θ2_dot, θ1_ddot, θ2_ddot = symbols('θ1 θ2 θ1_dot θ2_dot θ1_ddot θ2_ddot')
m1, m2, r1, r2, l1, l2, g, f1, f2, u = symbols('m1 m2 r1 r2 l1 l2 g f1 f2 u')

# Matrici definite nel problema
M = Matrix([
    [symbols('I1') + symbols('I2') + m1 * r1**2 + m2 * (l1**2 + r2**2) + 2 * m2 * l1 * r2 * cos(θ2),
     symbols('I2') + m2 * r2**2 + m2 * l1 * r2 * cos(θ2)],
    [symbols('I2') + m2 * r2**2 + m2 * l1 * r2 * cos(θ2),
     symbols('I2') + m2 * r2**2]
])

C = Matrix([
    [-m2 * l1 * r2 * θ2_dot * sin(θ2) * (θ2_dot + 2 * θ1_dot)],
    [m2 * l1 * r2 * sin(θ2) * θ1_dot**2]
])

G = Matrix([
    [g * (m1 * r1 + m2 * l1) * sin(θ1) + g * m2 * r2 * sin(θ1 + θ2)],
    [g * m2 * r2 * sin(θ1 + θ2)]
])

F = Matrix([
    [f1, 0],
    [0, f2]
])



# Matrice θ_dot
θ_dot = Matrix([θ1_dot, θ2_dot])

# Calcolo di M^-1 e soluzione
M_inv = M.inv()

rhs = Matrix([u, 0]) - C - F * θ_dot - G

θ_ddot_result = M_inv * rhs

Dynamics = Matrix([θ1_dot, θ2_dot, θ_ddot_result[0], θ_ddot_result[1]])

print("Dynamics:\n t1dot = ", Dynamics[0])
print("t2dot = ", Dynamics[1])
print("t1ddot = ", Dynamics[2])
print("t1dot = ", Dynamics[3])


# Variabili indipendenti per le derivate
variables = [θ1, θ2, θ1_dot, θ2_dot, u]

# Calcolo delle derivate per ciascuna variabile
θ1_ddot_derivatives = [θ_ddot_result[0].diff(var) for var in variables]
θ2_ddot_derivatives = [θ_ddot_result[1].diff(var) for var in variables]

# Risultati
print("Derivate di θ1_ddot:")
for var, derivative in zip(variables, θ1_ddot_derivatives):
    print(f"d(θ1_ddot)/d({var}) = {derivative}")

print("\nDerivate di θ2_ddot:")
for var, derivative in zip(variables, θ2_ddot_derivatives):
    print(f"d(θ2_ddot)/d({var}) = {derivative}")
