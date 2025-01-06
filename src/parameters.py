import numpy as np
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

t_i = 0
t_f = 20
dt = 1e-2

# Cost Function parameters
Qt = np.diag([1, 1, 2, 2])*1e3
Rt = np.diag([2])
QT = np.diag([1, 1, 5, 5])*1e4


# Armijo parameters
c = 0.5
beta = 0.7
Arm_plot = True
Arm_plot_every_k_iter = 4 
