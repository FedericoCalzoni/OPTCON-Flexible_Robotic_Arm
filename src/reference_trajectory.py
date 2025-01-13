import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import parameters as pm 
from newton_method import newton_method
import dynamics as dyn

def generate_trajectory(x_eq1, x_eq2, u_eq1, u_eq2, smooth_percentage=pm.smooth_percentage, t_f=pm.t_f, dt=pm.dt):
    """
    Generates a trajectory between two equilibrium points with smooth transitions.

    Args:
        x_eq1 (np.ndarray): Initial state equilibrium point.
        x_eq2 (np.ndarray): Final state equilibrium point.
        u_eq1 (np.ndarray): Initial control input equilibrium.
        u_eq2 (np.ndarray): Final control input equilibrium.
        smooth_percentage (float): Percentage of time allocated to smoothing.
        t_f (float): Final time for the trajectory.
        dt (float): Time step duration.

    Returns:
        tuple: x_reference (np.ndarray), u_reference (np.ndarray).
    """
    
    total_time_steps = int(t_f / dt)
    x_size = x_eq2.shape[0]
    
    # Initialize references
    x_reference = np.zeros((x_size, total_time_steps))
    u_reference = np.zeros((1, total_time_steps))

    # Create the cubic spline for the middle region
    t1 = total_time_steps/2 - total_time_steps*smooth_percentage / (2)
    t2 = total_time_steps/2 + total_time_steps*smooth_percentage / (2)
  
    for i in range(x_size):
        # Create a cubic spline to interpolate between x_eq1 and x_eq2
        if smooth_percentage != 0:
            spline = CubicSpline([t1, t2], np.vstack([x_eq1, x_eq2]), bc_type='clamped')
        for t in range(total_time_steps):
            if t <= t1:  # Before tf/4
                x_reference[i, t] = x_eq1[i]
            elif t > t2:  # After tf-(tf/4)
                x_reference[i, t] = x_eq2[i]
            else:  # Between tf/4 and tf-(tf/4)
                x_reference[i, t] = spline(t)[i] 
    
    for t in range(total_time_steps):
        G = dyn.compute_gravity(x_reference[2, t], x_reference[3, t])
        u_reference[0,t] = G[0]

    return x_reference, u_reference


def generate_smooth_trajectory(transition_width = pm.transition_width_task2):
    """
    Generates a smooth trajectory across multiple equilibrium points.

    Args:
        transition_width (int): Width of the transition between equilibria.

    Returns:
        tuple: x_reference (np.ndarray), u_reference (np.ndarray).
    """
    def calculate_dtheta1_dtheta2(x_reference):
      dtheta1 = np.diff(x_reference[2, :])/pm.dt
      dtheta2 = np.diff(x_reference[3, :])/pm.dt
      dtheta1 = np.append(dtheta1, dtheta1[-1])
      dtheta2 = np.append(dtheta2, dtheta2[-1])
      return dtheta1, dtheta2
    
    x_size = 4
    u_size = 1

    ### Compute each equilibrium point
    eq_list = [ -np.pi/2,  +np.pi/2, -np.pi/4,  + np.pi/4,     0]
    eq_state = ["normal", "normal", "normal", "normal", "normal"]
    equilibria = np.empty((len(eq_list), 3))
    x_eq = np.empty((len(eq_list), x_size))
    u_eq = np.empty((len(eq_list), u_size))

    K_eq = 44 # Derived by hand, approximated
    for i, eq in enumerate(eq_list):
        if eq_state[i]=="normal" or i >len(eq_state)-1:
            equilibria[i] = np.array([eq, -eq, K_eq * np.sin(eq)])
            x_eq[i], u_eq[i] = newton_method(equilibria[i][:,None])
        elif eq_state[i]=="upsidedown":
            equilibria[i] = np.array([eq, -np.pi-eq, K_eq * np.sin(eq)])
            x_eq[i], u_eq[i] = newton_method(equilibria[i][:,None])
    
    
    # Initialize references
    TT = pm.TT
    x_reference = np.zeros((x_size, TT))
    u_reference = np.zeros((u_size, TT))
    for t in range (TT):
        x_reference[:, t] = x_eq[0, :]
        u_reference[0, t] = u_eq[0, :]

    # Elaborate the reference to match the equilibria transition
    center = pm.transition_center
    for k in range(1, len(center)):
        t1 = int(center[k] - transition_width/ 2)
        t2 = int(center[k] + transition_width/ 2)
      
        for i in range(x_size):
            if transition_width != 0:
                spline = CubicSpline([t1, t2], np.vstack([x_eq[k-1, :], x_eq[k, :]]), bc_type='clamped')
            for t in range(t1, t2):
                x_reference[:, t] = spline(t)
            for t in range(t2, TT):
                x_reference[:, t] = x_reference[:, t-1]  
      
    dtheta1, dtheta2 = calculate_dtheta1_dtheta2(x_reference)

    for t in range(TT-1):
        # G = dyn.compute_gravity(x_reference[2, t], x_reference[3, t])
        # u_reference[0,t] = G[0]
        tau = dyn.inverse_dynamics(x_reference[:, t], x_reference[:,t+1])
        u_reference[0, t] = tau[0]
    u_reference[0, -1] = tau[0]

    x_reference[0,:] = dtheta1
    x_reference[1,:] = dtheta2
    
    return x_reference, u_reference


def plot_trajectory(x_reference, u_reference, t_f=pm.t_f, dt=pm.dt):
    """
    Plots the generated trajectory and control input.

    Args:
        x_reference (np.ndarray): State trajectory.
        u_reference (np.ndarray): Control input trajectory.
        t_f (float): Final time for the trajectory.
        dt (float): Time step duration.
    """

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))
    
    names = {0: r'$\dot \theta_1^{ref}$', 1: r'$\dot \theta_2^{ref}$', 2: r'$\theta_1^{ref}$', 3: r'$\theta_2^{ref}$'}
    colors = {0: 'm', 1: 'orange', 2: 'b', 3: 'g'}
    
    # Plot x_reference in a 2x2 grid
    for i in range(4):
        ax = fig.add_subplot(3, 2, i + 1)  # 3 rows, 2 columns, position i+1
        ax.plot(time, x_reference[i, :], color=colors[i], linewidth=2)
        ax.set_title(names[i])
        ax.set_ylabel('[rad/s]' if i < 2 else '[rad]')
        ax.grid(True)

    # Plot u_reference in a standalone plot
    ax = fig.add_subplot(3, 1, 3)  # Full-width plot at the bottom
    ax.plot(time, u_reference[0, :], color='r', linewidth=2)
    ax.set_title(r'$\tau_1^{ref}$')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[Nm]')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()