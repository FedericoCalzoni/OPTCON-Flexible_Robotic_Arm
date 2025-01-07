import numpy as np
from newton_method import newton_method
from dynamics import jacobian_gravity

z_0 = np.array([[np.pi/3], [-np.pi/3], [8]])

eq = newton_method(z_0, jacobian_gravity)

print("eq: ", eq)

