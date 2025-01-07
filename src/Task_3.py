import numpy as np
import matplotlib.pyplot as plt
import parameters as pm
from parameters import state_perturbation_percentage, input_perturbation_percentage
import dynamics as dyn


def LQR_system_regulator(x_gen, u_gen):
    print('\n\n\
        \t------------------------------------------\n \
        \t\tLaunching: LQR Tracker\n \
        \t------------------------------------------')
    
    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    TT = x_gen.shape[1]

    x_initial_guess = np.zeros((x_size, TT))
    u_initial_guess = np.zeros((u_size, TT))
    x_initial_guess[:, 0] = x_gen[:,0]*(1 + state_perturbation_percentage)
    u_initial_guess = u_gen*(1 + input_perturbation_percentage)

    # Initialize the perturberd system as the natural evolution of the system
    # without a proper regulation
    x_regulator = np.zeros((x_size, TT))
    u_regulator = np.zeros((u_size, TT))
    for t in range(TT):
        x_regulator[:, t+1] = dyn.dynamics()

    
    max_iterations = 1000

    l = np.zeros(max_iterations) # Cost function

    x_optimal = np.zeros((x_size, TT, max_iterations))
    u_optimal = np.zeros((u_size, TT, max_iterations))

    qt = np.zeros((x_size,TT-1))
    rt = np.zeros((u_size,TT-1))

    Lambda = np.zeros((4,TT))
    GradJ_u = np.zeros((u_size,TT-1))

    Qt_Star = np.zeros((x_size,x_size,TT-1))
    St_Star = np.zeros((u_size, x_size,TT-1))
    Rt_Star = np.zeros((u_size, u_size,TT-1))

    K_Star = np.zeros((u_size, x_size, TT-1, max_iterations))
    sigma_star = np.zeros((u_size, TT-1, max_iterations))
    delta_u = np.zeros((u_size, TT-1, max_iterations))

    A = np.zeros((x_size,x_size,TT-1))
    B = np.zeros((x_size,u_size,TT-1))





