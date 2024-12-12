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

# Visualizer parameters
t_i = 0
t_f = 10
dt = 1e-4

# Cost Function parameters
Qt = np.diag([1, 1, 1, 1])
Rt = np.diag([1, 0, 0, 0])
QT = np.diag([10000, 10000, 10000, 10000])