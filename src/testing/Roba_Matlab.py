import sympy as sp

# Define symbols
theta, omega, alpha = sp.symbols('theta omega alpha')  # Crank angle, angular velocity, angular acceleration
r, l, e = sp.symbols('r l e')  # Crank radius, rod length, offset
x = sp.symbols('x')  # Slider position

# Position equation
x_expr = r * sp.cos(theta) + sp.sqrt(l**2 - (r * sp.sin(theta) + e)**2)

# Velocity: differentiate position with respect to time
v_expr = sp.diff(x_expr, theta) * omega

# Acceleration: differentiate velocity with respect to time
a_expr = sp.diff(v_expr, theta) * omega + sp.diff(v_expr, omega) * alpha

# Simplify the results
x_simplified = sp.simplify(x_expr)
v_simplified = sp.simplify(v_expr)
a_simplified = sp.simplify(a_expr)

print(f"Formulax = {x_simplified}\nFormulav = {v_simplified}\nFormulaa = {a_simplified}")
