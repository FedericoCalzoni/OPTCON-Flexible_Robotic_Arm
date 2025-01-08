import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import parameters as pm
from parameters import t_f, dt
from parameters import smooth_percentage as smooth_period

def generate_trajectory(x_eq1, x_eq2, u_eq1, u_eq2):
    total_time_steps = int(pm.t_f / pm.dt)
    time = np.linspace(0, pm.t_f, total_time_steps)
    x_size = x_eq2.shape[0]
    
    # Initialize references
    x_reference = np.zeros((x_size, total_time_steps))
    u_reference = np.zeros((1, total_time_steps))

    # Create the cubic spline for the middle region
    t1 = pm.t_f / (2*pm.dt) - pm.t_f*smooth_period / (2*pm.dt)
    t2 = pm.t_f / (2*pm.dt) + pm.t_f*smooth_period / (2*pm.dt)
  
    for i in range(x_size):
        # Create a cubic spline to interpolate between x_eq1 and x_eq2
        if smooth_period != 0:
            spline = CubicSpline([t1, t2], np.vstack([x_eq1, x_eq2]), bc_type='clamped')
        for t in range(total_time_steps):
            if t <= t1:  # Before tf/4
                x_reference[i, t] = x_eq1[i]
            elif t > t2:  # After tf-(tf/4)
                x_reference[i, t] = x_eq2[i]
            else:  # Between tf/4 and tf-(tf/4)
                x_reference[i, t] = spline(t)[i] 

    if smooth_period != 0:
        spline = CubicSpline([t1, t2], np.vstack([u_eq1, u_eq2]), bc_type='clamped')
    for t in range(total_time_steps):
            if t <= t1:  # Before tf/4
                u_reference[:,t] = u_eq1
            elif t > t2:  # After tf-(tf/4)
                u_reference[:,t] = u_eq2
            else:  # Between tf/4 and tf-(tf/4)
                u_reference[:,t] = spline(t) 

    return x_reference, u_reference



def plot_trajectory(x_reference, u_reference):

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
