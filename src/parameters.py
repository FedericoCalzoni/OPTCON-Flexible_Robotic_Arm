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

def cost_matrices_computation(Qt_temp, Rt_temp, transition_width):
    # Output matrices
    Qt_final = np.zeros((4, 4, TT))
    Rt_final = np.zeros((1, 1, TT))

    # Time intervals
    phase1_end = int(TT / 3 - transition_width / 2)
    phase2_end = int(TT / 3 + transition_width / 2)
    phase3_end = int(2 * TT / 3 - transition_width / 2)
    phase4_end = int(2 * TT / 3 + transition_width / 2)

    for t in range(TT):
        if t < phase1_end:
            # Phase 1: Constant values
            Qt_final[:, :, t] = Qt_temp[:, :, 0]
            Rt_final[:, :, t] = Rt_temp[:, :, 0]
        elif phase1_end <= t < phase2_end:
            # Phase 2: Transition from [0] to [1]
            weight = smooth_transition(t, phase1_end, phase2_end)
            Qt_final[:, :, t] = (1 - weight) * Qt_temp[:, :, 0] + weight * Qt_temp[:, :, 1]
            Rt_final[:, :, t] = (1 - weight) * Rt_temp[:, :, 0] + weight * Rt_temp[:, :, 1]
        elif phase2_end <= t < phase3_end:
            # Phase 3: Constant values
            Qt_final[:, :, t] = Qt_temp[:, :, 1]
            Rt_final[:, :, t] = Rt_temp[:, :, 1]
        elif phase3_end <= t < phase4_end:
            # Phase 4: Transition from Qt_temp[1] to Qt_temp[2]
            weight = smooth_transition(t, phase3_end, phase4_end)
            Qt_final[:, :, t] = (1 - weight) * Qt_temp[:, :, 1] + weight * Qt_temp[:, :, 2]
            Rt_final[:, :, t] = (1 - weight) * Rt_temp[:, :, 1] + weight * Rt_temp[:, :, 2]
        else:
            # Phase 5: Constant values
            Qt_final[:, :, t] = Qt_temp[:, :, 2]
            Rt_final[:, :, t] = Rt_temp[:, :, 2]
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
t_f = 10
dt = 1e-3
TT = int((t_f - t_i)/dt)

######################################
##      Task 1 and 2 parameters     ##
######################################

optimal_trajectory_given = False    
smooth_percentage = 0
transition_width = TT/8
#Cost Function Parameters
# Initialize matrices
Qt_temp = np.zeros((4, 4, 3))
Rt_temp = np.zeros((1, 1, 3))

# Phase values
Qt_temp[:, :, 0] = np.diag([1, 1, 1, 1]) * 1e8
Rt_temp[:, :, 0] = np.diag([1]) * 1e2
Qt_temp[:, :, 1] = np.diag([1, 1, 1, 1]) * 0
Rt_temp[:, :, 1] = np.diag([3]) * 1e3
Qt_temp[:, :, 2] = np.diag([1, 1, 1, 1]) * 1e8
Rt_temp[:, :, 2] = np.diag([1]) * 1e2

# Assign final results
Qt, Rt = cost_matrices_computation(Qt_temp, Rt_temp, transition_width)
QT = Qt[:, :, -1]

# Armijo parameters
c = 0.5
beta = 0.7
Arm_plot = True
Arm_plot_every_k_iter = 5

Newton_Optcon_Plots = False
Newton_Plot_every_k_iterations = 3
plot_states_at_last_iteration = True


################################
##      Task 3 Parameters     ##
################################  

state_perturbation_percentage =0.1
input_perturbation_percentage =0.1
#Cost Function Parameters
smooth_percentage = 0
transition_width = TT/8
#Cost Function Parameters
# Initialize matrices
Qt_temp_reg = np.zeros((4, 4, 3))
Rt_temp_reg = np.zeros((1, 1, 3))

# Phase values
Qt_temp_reg[:, :, 0] = np.diag([1, 1, 1, 1]) * 1e8
Rt_temp_reg[:, :, 0] = np.diag([1]) * 1e2
Qt_temp_reg[:, :, 1] = np.diag([1, 1, 1, 1]) * 0
Rt_temp_reg[:, :, 1] = np.diag([3]) * 1e3
Qt_temp_reg[:, :, 2] = np.diag([1, 1, 1, 1]) * 1e8
Rt_temp_reg[:, :, 2] = np.diag([1]) * 1e2

Qt_reg, Rt_reg = cost_matrices_computation(Qt_temp_reg, Rt_temp_reg, transition_width)
QT_reg = Qt_reg[:, :, -1]

# Armijo parameters
c = 0.5
beta = 0.7


################################
##      Task 4 Parameters     ##
################################  

state_perturbation_percentage = -0.2

# MPC parameters
T_pred = 15
u_max = 60
u_min = -u_max
x_dtheta_max = 10
x_dtheta_min = -x_dtheta_max

# normal
x_theta1_max = 2*np.pi
x_theta1_min = -x_theta1_max
x_theta2_max = 2*np.pi
x_theta2_min = -x_theta2_max

Qt_temp = np.zeros((4, 4, 3))
Rt_temp = np.zeros((1, 1, 3))

# Phase values
Qt_temp[:, :, 0] = np.diag([1, 1, 100, 100])*1e2
Rt_temp[:, :, 0] = np.diag([1]) * 0
Qt_temp[:, :, 1] = np.diag([1, 10, 1, 1])*1e2
Rt_temp[:, :, 1] = np.diag([1]) * 0
Qt_temp[:, :, 2] = np.diag([10, 10, 1000, 10])*1e2
Rt_temp[:, :, 2] = np.diag([1]) * 0

# Assign final results
Qt, Rt = cost_matrices_computation(Qt_temp, Rt_temp, transition_width)
QT = Qt[:, :, -1]