import numpy as np

def smooth_transition(t, start, end):
    """
    Compute a smooth transition weight between 0 and 1 using a sigmoid function.

    Parameters:
        t (float): The time index.
        start (float): The start of the transition (t where weight ~ 0).
        end (float): The end of the transition (t where weight ~ 1).

    Returns:
        float: Transition weight between 0 and 1.
    """
    width = end - start
    return 1 / (1 + np.exp(-10 * (t - (start + width / 2)) / width))

# def cost_matrices_computation(Qt_temp, Rt_temp, transition_width):
#     # Output matrices
#     Qt_final = np.zeros((4, 4, TT))
#     Rt_final = np.zeros((1, 1, TT))

#     # Time intervals
#     phase1_end = int(TT / 3 - transition_width / 2)
#     phase2_end = int(TT / 3 + transition_width / 2)
#     phase3_end = int(2 * TT / 3 - transition_width / 2)
#     phase4_end = int(2 * TT / 3 + transition_width / 2)

#     for t in range(TT):
#         if t < phase1_end:
#             # Phase 1: Constant values
#             Qt_final[:, :, t] = Qt_temp[:, :, 0]
#             Rt_final[:, :, t] = Rt_temp[:, :, 0]
#         elif phase1_end <= t < phase2_end:
#             # Phase 2: Transition from [0] to [1]
#             weight = smooth_transition(t, phase1_end, phase2_end)
#             Qt_final[:, :, t] = (1 - weight) * Qt_temp[:, :, 0] + weight * Qt_temp[:, :, 1]
#             Rt_final[:, :, t] = (1 - weight) * Rt_temp[:, :, 0] + weight * Rt_temp[:, :, 1]
#         elif phase2_end <= t < phase3_end:
#             # Phase 3: Constant values
#             Qt_final[:, :, t] = Qt_temp[:, :, 1]
#             Rt_final[:, :, t] = Rt_temp[:, :, 1]
#         elif phase3_end <= t < phase4_end:
#             # Phase 4: Transition from Qt_temp[1] to Qt_temp[2]
#             weight = smooth_transition(t, phase3_end, phase4_end)
#             Qt_final[:, :, t] = (1 - weight) * Qt_temp[:, :, 1] + weight * Qt_temp[:, :, 2]
#             Rt_final[:, :, t] = (1 - weight) * Rt_temp[:, :, 1] + weight * Rt_temp[:, :, 2]
#         else:
#             # Phase 5: Constant values
#             Qt_final[:, :, t] = Qt_temp[:, :, 2]
#             Rt_final[:, :, t] = Rt_temp[:, :, 2]
#     return Qt_final, Rt_final

def compute_intervals(total_time, num_divisions, transition_width, rest_start, rest_end):
    """
    Compute the start and end indices for each phase and transition.
    """
    transition_time = total_time - rest_start - rest_end
    interval_length = transition_time / num_divisions
    intervals = []
    for i in range(num_divisions):
        start = int(i * interval_length)+rest_start
        end = int((i + 1) * interval_length)+rest_start
        transition_end = max(start, end - int(transition_width / 2))
        transition_middle = int((start + end) / 2)
        transition_start = min(end, start + int(transition_width / 2))
        intervals.append((start, transition_start, transition_middle, transition_end, end))
    # print(intervals)
    return intervals

def cost_matrices_computation(Qt_temp, Rt_temp, total_time, num_divisions, transition_width):
    """
    Compute the cost matrices for given divisions and transitions.
    """
    num_states = Qt_temp.shape[0]
    num_controls = Rt_temp.shape[0]
    rest_start = 2000
    rest_end = 2000
    Qt_final = np.zeros((num_states, num_states, total_time))
    Rt_final = np.zeros((num_controls, num_controls, total_time))

    for t in range(total_time):
        Qt_final[:,:,t] = Qt_temp[:, :, 0]
        Rt_final[:,:,t] = Rt_temp[:, :, 0]

    intervals = compute_intervals(total_time, num_divisions, transition_width, rest_start, rest_end)

    for i, (start, transition_start, transition_middle, transition_end, end) in enumerate(intervals):
        for t in range(start, end):
            if t < start:  # Skip earlier phases
                continue
            if start <= t < transition_start:  # Constant phase
                Qt_final[:, :, t] = Qt_temp[:, :, 0]
                Rt_final[:, :, t] = Rt_temp[:, :, 0]
            elif transition_start <= t < transition_middle:  # Transition phase
                weight = smooth_transition(t, transition_start, transition_middle)
                Qt_final[:, :, t] = (1 - weight) * Qt_temp[:, :, 0] + weight * Qt_temp[:, :, 1]
                Rt_final[:, :, t] = (1 - weight) * Rt_temp[:, :, 0] + weight * Rt_temp[:, :, 1]
            elif transition_middle <= t < transition_end:  # Transition phase
                weight = smooth_transition(t, transition_middle, transition_end)
                Qt_final[:, :, t] = (1 - weight) * Qt_temp[:, :, 1] + weight * Qt_temp[:, :, 0]
                Rt_final[:, :, t] = (1 - weight) * Rt_temp[:, :, 1] + weight * Rt_temp[:, :, 0]
            elif transition_end <= t < end:  # Constant phase again
                Qt_final[:, :, t] = Qt_temp[:, :, 0]
                Rt_final[:, :, t] = Rt_temp[:, :, 0]
            elif t >= end:  # Move to the next phase
                break

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
LQR_trajectory_given = False 
MPC_trajectory_given = True  

######################################
##      Task 1 and 2 parameters     ##
######################################

smooth_percentage = 0.5
transition_width = 700
divisions = 4

#Cost Function Parameters
# Initialize matrices
Qt_temp = np.zeros((4, 4, 2))
Rt_temp = np.zeros((1, 1, 2))

# Phase values
# Assign values

# Qt_temp[:, :, 0] = np.diag([80000, 100000, 5000000, 300000])   # Constant phase
# Rt_temp[:, :, 0] = np.diag([500])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([20, 20, 1000, 0])                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([500]) 

# Qt_temp[:, :, 0] = np.diag([60162, 130281, 4999904, 300011])   # Constant phase
# Rt_temp[:, :, 0] = np.diag([55.8])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([1e-8, 1e-8, 1.54, 1e-8])                   # Transition phase
# Rt_temp[:, :, 1] = np.diag([16.54]) 

# Qt_temp[:, :, 0] = np.diag([1000, 1000, 500000, 30000])   # Constant phase
# Rt_temp[:, :, 0] = np.diag([50])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([20, 20, 20, 0])                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([50]) 

# Qt_temp[:, :, 0] = np.diag([60000, 130000, 5000000, 100000])   # Constant phase
# Rt_temp[:, :, 0] = np.diag([80])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([60, 13000, 100, 0])                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([80]) 

# Qt_temp[:, :, 0] = np.diag([59158.74622544, 129245.63949172, 4999870.45305163, 299992.96147837])   # Constant phase
# Rt_temp[:, :, 0] = np.diag([75.83220289])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([61.26095827, 26.20654226, 47.82525748, 26.32315921])                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([14.86213965]) 

# Qt_temp[:, :, 0] = np.diag([59158.74622544, 129245.63949172, 4999870.45305163, 299992.96147837])   # Constant phase
# Rt_temp[:, :, 0] = np.diag([75.83220289])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([0, 0, 0, 0])                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([7.86213965]) 

# Qt_temp[:, :, 0] = np.diag([59158.74622544, 129245.63949172, 4999870.45305163, 299992.96147837])   # Constant phase
# Rt_temp[:, :, 0] = np.diag([75.83220289])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([61.26095827, 26.20654226, 47.82525748, 26.32315921])                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([14.86213965]) 

# # Qt_temp[:, :, 0] = np.diag([1, 1, 1, 1]) * 1e8   # Constant phase
# Rt_temp[:, :, 0] = np.diag([1]) * 1e2                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([1,1,1,1]) * 0                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([3]) * 1e3 


# DA INCORNICIARE REFERENCE=111, OUTPUT117
Qt_temp[:, :, 0] = np.diag([9.99854091, 999.331936, 4909.934141, 10.0133061]) *1e7  # Constant phase
Rt_temp[:, :, 0] = np.diag([270.5502102])                                 # Constant phase
Qt_temp[:, :, 1] = np.diag([0, 0.620181367, 0, 6.79470332])                   # Transition phase
Rt_temp[:, :, 1] = np.diag([296.05527468])

# # DA TATUARE
# Qt_temp[:, :, 0] = np.diag([1.00001241, 99.9331399, 490.9935389, 1.00096131]) *1e8  # Constant phase
# Rt_temp[:, :, 0] = np.diag([265.5499998])                                 # Constant phase
# Qt_temp[:, :, 1] = np.diag([1.02688787, 0.00000001, 3.33155371, 4.92487656]) * 0                  # Transition phase
# Rt_temp[:, :, 1] = np.diag([299.40677386])




# Assign final results
Qt, Rt = cost_matrices_computation(Qt_temp, Rt_temp, TT, divisions, transition_width)
QT = Qt[:, :, -1]

# print(Qt.shape, Rt.shape)

# from matplotlib import pyplot as plt
# for i in range(4):
#     plt.plot(Qt[i, i, :])
# plt.plot(Rt[0, 0, :])
# plt.show()


# Armijo parameters
c = 0.5
beta = 0.7
Arm_plot = False
Arm_plot_every_k_iter = 2

Newton_Optcon_Plots = False
Newton_Plot_every_k_iterations = 2
plot_states_at_last_iteration = False

################################
##      Task 3 Parameters     ##
################################  

state_perturbation_percentage = 0
affine_perturbation = 0

# Cost Function Parameters
Qt_temp_reg = np.zeros((4, 4, 2))
Rt_temp_reg = np.zeros((1, 1, 2))

Qt_temp_reg[:, :, 0] = np.diag([1, 1, 1, 1]) * 1e7
Rt_temp_reg[:, :, 0] = np.diag([5]) * 1e1
Qt_temp_reg[:, :, 1] = np.diag([1, 1, 1, 1]) * 1e2
Rt_temp_reg[:, :, 1] = np.diag([1]) * 1e0

# Qt_temp_reg[:, :, 0] = np.diag([0, 0, 4909.934141, 10.0133061]) *1e5  # Constant phase
# Rt_temp_reg[:, :, 0] = np.diag([27.5502102])                                 # Constant phase
# Qt_temp_reg[:, :, 1] = np.diag([0, 0.620181367, 0, 6.79470332])                   # Transition phase
# Rt_temp_reg[:, :, 1] = np.diag([2.05527468])

Qt_reg, Rt_reg = cost_matrices_computation(Qt_temp_reg, Rt_temp_reg, TT, divisions, transition_width)
QT_reg = Qt_reg[:, :, -1]

################################
##      Task 4 Parameters     ##
################################  

state_perturbation_percentage = -0.2

# MPC parameters
T_pred = 5
u_max = 60
u_min = -u_max
x_dtheta_max = 100
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
Qt_temp_MPC[:, :, 0] = np.diag([1, 1, 10000, 10000])
Rt_temp_MPC[:, :, 0] = np.diag([1]) * 0
Qt_temp_MPC[:, :, 1] = np.diag([0, 0, 1, 1])*1e2
Rt_temp_MPC[:, :, 1] = np.diag([1]) * 0
Qt_temp_MPC[:, :, 2] = np.diag([1, 1, 10000, 10000])
Rt_temp_MPC[:, :, 2] = np.diag([1]) * 0


# Assign final results
Qt_MPC, Rt_MPC = cost_matrices_computation(Qt_temp_MPC, Rt_temp_MPC, TT, divisions, transition_width)
QT_MPC = Qt_MPC[:, :, -1]