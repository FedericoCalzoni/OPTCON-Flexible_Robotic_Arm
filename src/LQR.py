import numpy as np
from numpy.linalg import inv 
import matplotlib.pyplot as plt
import parameters as pm
from parameters import state_perturbation_percentage, affine_perturbation
import dynamics as dyn
import costTask3 as cost


def LQR_system_regulator(x_gen, u_gen):
    print('\n\n\
        \t------------------------------------------\n \
        \t\tLaunching: LQR Tracker\n \
        \t------------------------------------------')
    
    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    TT = x_gen.shape[1]

    x_regulator = np.zeros((x_size, TT))
    u_regulator = np.zeros((u_size, TT))
    x_natural_evolution = np.zeros((x_size, TT))
    x_evolution_after_LQR = np.zeros((x_size, TT))

    x_regulator[:, 0]         = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_natural_evolution [:,0] = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_evolution_after_LQR[:,0]= x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation

    u_regulator = u_gen

    delta_x = x_regulator - x_gen
    delta_u = u_regulator - u_gen

    # Initialize the perturberd system as the natural evolution of the system
    # without a proper regulation
    for t in range(TT-1):
        x_natural_evolution[:, t+1] = dyn.dynamics(x_natural_evolution[:, t], u_regulator[:,t])


    Qt = np.zeros((x_size, x_size,TT-1))
    Rt = np.zeros((u_size, u_size,TT-1))

    K_Star = np.zeros((u_size, x_size, TT-1))

    A = np.zeros((x_size,x_size,TT-1))
    B = np.zeros((x_size,u_size,TT-1))

    # Calcolo le jacobiane di dinamica e costo
    for t in range(TT-1):
        A[:,:,t] = dyn.jacobian_x_new_wrt_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = dyn.jacobian_x_new_wrt_u(x_gen[:,t])
        Qt[:,:,t] = cost.hessian1_J(t)           
        Rt[:,:,t] = cost.hessian2_J(t)              
    QT = cost.hessian_terminal_cost()

    K_Star = LQR_solver(A, B, Qt, Rt, QT)

    for t in range(TT-1):
        delta_u[:, t]  = K_Star[:,:,t] @ delta_x[:, t]
        delta_x[:, t+1]= A[:,:,t] @ delta_x[:,t] + B[:,:,t] @ delta_u[:, t]
    
    for t in range(TT-1):
        u_regulator[:,t] = u_gen[:,t] + delta_u[:,t]
        x_regulator[:,t] = x_gen[:,t] + delta_x[:,t]
        x_evolution_after_LQR[:,t+1] = dyn.dynamics(x_evolution_after_LQR[:,t], u_regulator[:,t])

    plt.figure()
    for i in range(x_size):
        plt.plot(x_evolution_after_LQR[i, :], color = 'red', label = f'x[{i}]')
    plt.plot(u_regulator[0,:], color = 'blue', label = 'u_regulator')
    plt.title("System Evolution with Real Dynamics and LQRegulated input")
    plt.legend()
    plt.grid()
    plt.show()

    # NOTA: con delta_x e delta_u, l'LQR è perfettamente in grado di annullare la distanza dalla reference
    
    plt.figure()
    for i in range(x_size):
        plt.plot(delta_x[i, :], color = 'red', label = r'$\Delta$' f'x[{i}]')
    plt.plot(delta_u[0,:], color = 'blue', label = r'$\Delta$' 'u')
    plt.title("LQR Residuals evolution")
    plt.legend()
    plt.grid()
    plt.show()

    return x_evolution_after_LQR, u_regulator


def LQR_solver(A, B, Qt_Star, Rt_Star, QT_Star):
    x_size = A.shape[0]
    u_size = B.shape[1]
    TT = A.shape[2]+1
    
    delta_x = np.zeros((x_size,TT))

    P = np.zeros((x_size,x_size,TT))
    Pt = np.zeros((x_size,x_size))
    Ptt= np.zeros((x_size,x_size))
    
    K = np.zeros((u_size,x_size,TT-1))
    Kt= np.zeros((u_size,x_size))

    ######### Solve the Riccati Equation [S6C4]
    P[:,:,-1] = QT_Star

    for t in reversed(range(TT-1)):
        At  = A[:,:,t]
        Bt  = B[:,:,t]
        Qt  = Qt_Star[:,:,t]
        Rt  = Rt_Star[:,:,t]
        Ptt = P[:,:,t+1]

        temp = (Rt + Bt.T @ Ptt @ Bt)
        inv_temp = inv(temp)
        Kt =-inv_temp @ (Bt.T @ Ptt @ At)
        Pt = At.T @ Ptt @ At + At.T@ Ptt @ Bt @ Kt + Qt

        K[:,:,t] = Kt
        P[:,:,t] = Pt 
    return K

def plot_LQR_error(x_LQR, x_gen):
    plt.figure()
    for i in range(4):
        plt.plot(x_LQR[i, :], color = 'red', label = f'x_LQR[{0}]' )
        plt.plot(x_gen[i, :], color = 'blue', label = f'x_gen[{0}]' )
    plt.title('Optimal Trajectory VS LQR Trajectory')
    plt.grid()
    plt.legend()
    plt.show(block = True)




