import numpy as np
import matplotlib.pyplot as plt


def smooth_transition(start_value, end_value, start_ind, end_ind):
    """
    Compute a smooth transition weight between 0 and 1 using a poly 3 function.

    Parameters:
        t (float): The time index.

    Returns:
        Transition evolution
    """
    delta_ind = end_ind - start_ind

    t_sim = 1/dt

    tf = delta_ind/t_sim


    A =np.array([[1,  0,     0,     0],
                [1, tf, tf**2, tf**3],
                [0, 1,      0,     0],
                [0, 1,   2*tf, 3*tf**2]])

    b = np.array([[start_value],
                [end_value],
                [0],
                [0]])

    coefficients = np.linalg.solve(A, b)

    a1 = coefficients[0]
    a2 = coefficients[1]
    a3 = coefficients[2]
    a4 = coefficients[3]

    evolution = np.zeros((delta_ind, 1))
    for t in range(delta_ind):
        evolution[t] = a1 + a2*t/t_sim + a3*(t/t_sim)**2 + a4*(t/t_sim)**3
    return  evolution.flatten()

def cost_matrices_computation(Qt_temp, Rt_temp, TT, num_divisions, transition_width):
    """
    Compute the cost matrices for given divisions and transitions.
    """
    x_size = Qt_temp.shape[0]
    u_size = Rt_temp.shape[0]

    Qt_final = np.zeros((x_size, x_size, TT))
    Rt_final = np.zeros((u_size, u_size, TT))
    for t in range(TT):
        Qt_final[:,:,t] = Qt_temp[:,:,0]
        Rt_final[:,:,t] = Rt_temp[:,:,0]

    num_stable_states = num_divisions + 1
    transition_center = [0]
    for i in range(1,  num_stable_states):
        transition_center.append(i * TT/num_stable_states)


    for i in range(1, num_stable_states):
        # Assuming a smooth transition takes a sixth of the transition time 
        transition_start = int(transition_center[i] - transition_width/2)
        transition_end =   int(transition_center[i] - transition_width/4)
        for j in range(x_size):
            Qt_final[j,j, transition_start : transition_end] = \
                smooth_transition(Qt_temp[j,j,0], Qt_temp[j,j,1], transition_start, transition_end)
            Qt_final[j,j, transition_end:TT] = Qt_temp[j,j,1]

        Rt_final[0,0,transition_start : transition_end] = \
            smooth_transition(Rt_temp[0,0,0], Rt_temp[0,0,1], transition_start, transition_end)
        Rt_final[0,0, transition_end:TT] = Rt_temp[0,0,1]
           
        
        transition_start = int(transition_center[i] + transition_width/3)
        transition_end =   int(transition_center[i] + transition_width/2)
        for j in range(x_size):
            Qt_final[j,j, transition_start : transition_end] = \
                smooth_transition(Qt_temp[j,j,1], Qt_temp[j,j,0], transition_start, transition_end)
            Qt_final[j,j, transition_end:TT] = Qt_temp[j,j,0]
        Rt_final[0,0,transition_start : transition_end] = \
            smooth_transition(Rt_temp[0,0,1], Rt_temp[0,0,0], transition_start, transition_end)
        Rt_final[0,0, transition_end:TT] = Rt_temp[0,0,0]  
    return Qt_final, Rt_final

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
t_f = 36
dt = 1e-3
TT = int((t_f - t_i)/dt)

optimal_trajectory_given = False
initial_guess_given = False
LQR_trajectory_given = False 
MPC_trajectory_given = True  

######################################
##      Task 1 and 2 parameters     ##
######################################

smooth_percentage = 0.5
divisions = 4
transition_width = int(TT/6)
num_stable_states = divisions + 1
transition_center = [0]
for i in range(1,  num_stable_states):
    transition_center.append(i * TT/num_stable_states)


#Cost Function Parameters
# Initialize matrices
Qt_temp = np.zeros((4, 4, 2))
Rt_temp = np.zeros((1, 1, 2))

# Phase values
# Assign values

# DA INCORNICIARE REFERENCE=111, OUTPUT117
# Qt_temp[:, :, 0] = np.diag([9.99854091, 999.331936, 4909.934141, 10.0133061]) *1e7  # Constant phase
# Rt_temp[:, :, 0] = np.diag([270.5502102])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([0, 0.620181367, 0, 6.79470332])                   # Transition phase
# Rt_temp[:, :, 1] = np.diag([296.05527468])

# # DA TATUARE
# Qt_temp[:, :, 0] = np.diag([1.00001241, 99.9331399, 490.9935389, 1.00096131]) *1e8  # Constant phase
# Rt_temp[:, :, 0] = np.diag([265.5499998])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([1.02688787, 0.00000001, 3.33155371, 4.92487656]) * 0                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([299.40677386])

# TI Voglio Beneh
# Qt_temp[:, :, 0] = np.diag([10, 10,  8, 8]) *1e6  # Constant phase
# Rt_temp[:, :, 0] = np.diag([1])*1e3                            # Constant phase
# Qt_temp[:, :, 1] = np.diag([0.1,0.1,1,1])*1e0                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([1])*1e3

Qt_temp[:, :, 0] = np.diag([10, 10,  8, 8]) *1e6  # Constant phase
Rt_temp[:, :, 0] = np.diag([1])*1e3                            # Constant phase
Qt_temp[:, :, 1] = np.diag([0.1,0.1,1,1])*1e0                  # Transition phase
Rt_temp[:, :, 1] = np.diag([1])*1e3


# Assign final results
Qt, Rt = cost_matrices_computation(Qt_temp, Rt_temp, TT, divisions, transition_width)
QT = Qt[:, :, -1]

# from matplotlib import pyplot as plt
# for i in range(4):
#     plt.plot(Qt[i, i, :])
# plt.plot(Rt[0, 0, :])
# plt.show()

# Armijo parameters
c = 0.5
beta = 0.7
arm_max_iter = 100
Arm_plot = False
Arm_plot_every_k_iter = 2

Newton_Optcon_Plots = False
Newton_Plot_every_k_iterations = 5
plot_states_at_last_iteration = False

################################
##      Task 3 Parameters     ##
################################  

state_perturbation_percentage = 0.00
affine_perturbation = 0.0

# Cost Function Parameters
Qt_temp_reg = np.zeros((4, 4, 2))
Rt_temp_reg = np.zeros((1, 1, 2))

# Qt_temp_reg[:, :, 0] = np.diag([9.99854091, 999.331936, 4909.934141, 10.0133061]) *1e7  # Constant phase
# Rt_temp_reg[:, :, 0] = np.diag([270.5502102])                                     # Constant phase
# Qt_temp_reg[:, :, 1] = np.diag([0, 0.620181367, 0, 6.79470332])                   # Transition phase
# Rt_temp_reg[:, :, 1] = np.diag([296.05527468])

Qt_temp_reg[:, :, 0] = np.diag([10, 10,  10, 10]) *1e2               # Constant phase
Rt_temp_reg[:, :, 0] = np.diag([10])*1e2                            # Constant phase
Qt_temp_reg[:, :, 1] = np.diag([10,10,10,10])*1e2                  # Transition phase
Rt_temp_reg[:, :, 1] = np.diag([10])*1e2

Qt_reg, Rt_reg = cost_matrices_computation(Qt_temp_reg, Rt_temp_reg, TT, divisions, transition_width)
QT_reg = Qt_reg[:, :, -1]

################################
##      Task 4 Parameters     ##
################################  

state_initial_perturbation = 0.02
noise_sensor = 0.05
noise_actuator = 0.05

# MPC parameters
T_pred = 5
u_max = 60
u_min = -u_max
x_dtheta_max = 10
x_dtheta_min = -x_dtheta_max

# normal
x_theta1_max = 2*np.pi
x_theta1_min = -x_theta1_max
x_theta2_max = 2*np.pi
x_theta2_min = -x_theta2_max

Qt_temp_MPC = np.zeros((4, 4, 3))
Rt_temp_MPC = np.zeros((1, 1, 3))

# # Normal
# Qt_temp_MPC[:, :, 0] = np.diag([1, 1, 100, 100])*1e2
# Rt_temp_MPC[:, :, 0] = np.diag([1]) * 0
# Qt_temp_MPC[:, :, 1] = np.diag([1, 10, 1, 1])*1e2
# Rt_temp_MPC[:, :, 1] = np.diag([1]) * 0
# Qt_temp_MPC[:, :, 2] = np.diag([10, 10, 1000, 10])*1e2
# Rt_temp_MPC[:, :, 2] = np.diag([1]) * 0

# Loop
# Qt_temp_MPC[:, :, 0] = np.diag([1, 1, 10000, 10000])
# Rt_temp_MPC[:, :, 0] = np.diag([1]) * 0
# Qt_temp_MPC[:, :, 1] = np.diag([0, 0, 1, 1])*1e2
# Rt_temp_MPC[:, :, 1] = np.diag([1]) * 0
# Qt_temp_MPC[:, :, 2] = np.diag([1, 1, 10000, 10000])
# Rt_temp_MPC[:, :, 2] = np.diag([1]) * 0

# # No noise
# Qt_temp_MPC[:, :, 0] = np.diag([1, 1, 4000, 4000])  * (1/1.653676929332829) # Constant phase
# Rt_temp_MPC[:, :, 0] = np.diag([0.001]) * (1/44.78666325774839)                                 # Constant phase
# Qt_temp_MPC[:, :, 1] = np.diag([1, 1, 1000, 1000])                   # Transition phase
# Rt_temp_MPC[:, :, 1] = np.diag([0.001]) * (1/44.78666325774839)

# # with noise (-0.2, 0.05, 0.05)
Qt_temp_MPC[:, :, 0] = np.diag([1, 1, 5000, 5000])  * (1/1.653676929332829) # Constant phase
Rt_temp_MPC[:, :, 0] = np.diag([0.001]) * (1/44.78666325774839)                                 # Constant phase
Qt_temp_MPC[:, :, 1] = np.diag([1, 1, 1000, 1000])                   # Transition phase
Rt_temp_MPC[:, :, 1] = np.diag([0.001]) * (1/44.78666325774839)

# # with extreme noise
# Qt_temp_MPC[:, :, 0] = np.diag([100000, 100000, 2000000000, 5000000])  * (1/1.653676929332829) # Constant phase
# Rt_temp_MPC[:, :, 0] = np.diag([100]) * (1/44.78666325774839)                                 # Constant phase
# Qt_temp_MPC[:, :, 1] = np.diag([100000, 100000, 2000000000, 5000000])                   # Transition phase
# Rt_temp_MPC[:, :, 1] = np.diag([100]) * (1/44.78666325774839)

# Assign final results
Qt_MPC, Rt_MPC = cost_matrices_computation(Qt_temp_MPC, Rt_temp_MPC, TT, divisions, transition_width)
QT_MPC = Qt_MPC[:, :, -1]