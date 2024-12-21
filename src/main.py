import dynamics as dyn
from dynamics import jacobian
from visualizer import animate_double_pendulum as anim
import numpy as np
from parameters import L1, L2, t_i, t_f, dt
from newton_method import newton_method
import cost
from gradient_method import gradient_method
from reference_trajectory import generate_trajectory

def main():
    """
    Main function to simulate the dynamics and visualize the results.
    """
    # Initial state and input
    x_0 = np.array([[0], [0], [np.pi/6], [np.pi]])  # Initial state (dtheta1, dtheta2, theta1, theta2)
    u_0 = np.array([[10], [0], [0], [0]])  # Input (tau1, tau2 , - ,  - )
    x_size = 4
    u_size = 4
    
    # Find equilibria
    
    z_0_eq1 = np.array([[-np.pi/3], [+np.pi/3], [-8]])
    z_0_eq2 = np.array([[np.pi/3], [-np.pi/3], [8]])
    equilibria_1 = newton_method(z_0_eq1, jacobian)
    equilibria_2 = newton_method(z_0_eq2, jacobian)
    eq1_0 = equilibria_1[0].item()
    eq1_1 = equilibria_1[1].item()
    eq1_2 = equilibria_1[2].item()
    eq2_0 = equilibria_2[0].item()
    eq2_1 = equilibria_2[1].item()
    eq2_2 = equilibria_2[2].item()
    print("eq: ", equilibria_2)
    
    x_0 = np.array([[0], [0], [eq1_0], [eq1_1]])
    u_0 = np.array([[eq1_2], [0], [0], [0]])
   


    # Compute dynamics for each time step
    time_intervals = int((t_f - t_i) / dt + 1)
    x_history = [x_0]
    print("Computing dynamics...")
    for i in range(time_intervals):
        x_history.append(dyn.dynamics(x_history[i], u_0, dt)[0])
    
    # x_e = np.zeros((time_intervals, 4))
    # for i in range(1, time_intervals):
    #     x_e[i] = (x_history[i]).flatten() -(x_history[i-1]).flatten()
    
    # plt.plot(x_e)
    
    
    ###### TASK 1 - J computation trials - Mike
    x_f = np.array([[0], [0], [eq2_0], [eq2_1]])
    u_f = np.array([[eq2_2], [0], [0], [0]])

    x_ref = np.zeros([4, time_intervals])
    u_ref = np.zeros([4, time_intervals])

    # Build the reference curve which must be used to compute the cost function
    for i in range (int(time_intervals / 2)):
        x_ref[:, i] = x_0.flatten()
        u_ref[:, i] = u_0.flatten()

    for i in range (int(time_intervals/2) , time_intervals):
        x_ref[:, i] = x_f.flatten()
        u_ref[:, i] = u_f.flatten()

    # Build the x and u trajectory which permanently stays on the first equilibrium 
    # (from t =  0 up to t = T) as an initial guess for the newton method for optimal control
    # Fede ha già chiamato x_trajectory come x_history. Sono la stessa cosa ma volevo avere
    # un'istanza differente per poter debuggare senza generare potenziali errori l'un l'altro.
    x_trajectory = np.zeros([4, time_intervals])
    u_trajectory = np.zeros([4, time_intervals]) 
    for i in range (time_intervals):
        x_trajectory[:, i] = x_0.flatten()
        u_trajectory[:, i] = u_0.flatten()

    J = cost.J_Function(x_trajectory, u_trajectory, x_ref, u_ref)
    print(time_intervals)
    print(J)
    #########
    
    # compute reference trajectory
    tf = 10
    x_eq1 = np.array([0.0, 0.0, 0.0, 0.0])
    x_eq2 = np.array([0.0, 0.0, 10.0, 5.0])
    k_eq = 2.0
    
    x_reference, u_reference = generate_trajectory(tf, x_eq1, x_eq2, k_eq)
    
    T = x_reference.shape[1]
    
    # Compute optimal trajectory
    # Initial guess
    x_init = np.zeros((x_size, T))
    u_init = np.zeros((u_size, T))
    x_optimal, u_optimal, J, lmbd = gradient_method(x_init, u_init, x_reference, u_reference)



    # Convert state history to a matrix
    matrix_x_history = np.hstack(x_history)

    # Visualize the simulation
    anim(matrix_x_history.T, L1, L2, frame_skip=1)

if __name__ == "__main__":
    main()
