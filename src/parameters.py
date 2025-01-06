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
t_f = 5
dt = 1e-2


# Cost Function parameters
#Qt = np.diag([1, 1, 10, 10])
#Rt = np.diag([1])
#QT = np.diag([10, 10, 15, 15])
Rt = np.diag([1])
Qt = np.diag([1, 1, 1, 1])
QT = np.diag([50, 50, 50, 50])


# Armijo parameters
c = 0.5
beta = 0.7
Arm_plot = False
Arm_plot_every_k_iter = 5

Newton_Optcon_Plots = False
Newton_Plot_every_k_iterations = 3
plot_states_at_last_iteration = True

# MPC parameters
T_pred = 5
umax=50
umin=-umax