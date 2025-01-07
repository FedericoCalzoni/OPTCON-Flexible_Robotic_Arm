from sympy import *
I1, I2, M1, M2, R1, R2, L1, G, theta1, theta2 = symbols('I1 I2 M1 M2 R1 R2 L1 G theta1 theta2')
expr = -I1*I2 + I1*M2*R2**2 + I2*L1**2*M2 + I2*M1*R1**2 - L1**2*M2**2*R2**2*cos(2*theta2) + L1**2*M2**2*R2**2 + M1*M2*R1**2*R2**2 
expr /= (G*M2*R2*sin(theta1 + theta2) + L1*M2*R2*theta1**2*sin(theta2)) 
expr /= (I1 + I2 + L1**2*M2 + 2*L1*M2*R2*cos(theta2) + M1*R1**2 + M2*R2**2)
derivative = diff(expr, theta2)
print(derivative)