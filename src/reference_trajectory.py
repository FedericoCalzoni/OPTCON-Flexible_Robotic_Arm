import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import parameters as param

def generate_trajectory(tf, x_eq1, x_eq2, u_eq1, u_eq2, smooth_period=0):
    total_time_steps = int(tf / param.dt)
    time = np.linspace(0, tf, total_time_steps)
    x_size = x_eq2.shape[0]
    
    # Initialize references
    x_reference = np.zeros((x_size, total_time_steps))
    u_reference = np.zeros((1, total_time_steps))

    # Create the cubic spline for the middle region
    t1 = tf / (2*param.dt) - tf*smooth_period / (2*param.dt)
    t2 = tf / (2*param.dt) + tf*smooth_period / (2*param.dt)
  
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

# TODO: Implement this function
# def plot_trajectory(x_reference, u_reference, dt=1e-3):

# TODO: remove this test function
# Test function and plotting
def test_generate_trajectory():
    # Define test parameters
    tf = 4.0  # Total time duration
    x_eq1 = np.array([0.0, 0.0]) 
    x_eq2 = np.array([10.0, 5.0])
    k_eq = 2.0
    dt = 0.01 
    smooth_period = 0.5 

    # Generate smooth trajectory
    x_ref_smooth, u_ref_smooth = generate_trajectory(tf,x_eq1, x_eq2, k_eq, smooth_period, dt)
    # Generate non-smooth trajectory
    x_ref_linear, u_ref_linear = generate_trajectory(tf, x_eq1, x_eq2, k_eq, 0, dt)
    
    time = np.linspace(0, tf, int(tf / dt))
    
    # Plot smooth vs non-smooth trajectories
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Trajectory Comparisons")

    for i in range(x_eq2.shape[0]):
        # Plot x_reference
        axes[0, i].plot(time, x_ref_smooth[i, :], label="Smooth", linewidth=2)
        axes[0, i].plot(time, x_ref_linear[i, :], label="Linear", linestyle="--", linewidth=2)
        axes[0, i].set_title(f"x_reference (Dimension {i+1})")
        axes[0, i].set_xlabel("Time (s)")
        axes[0, i].set_ylabel("x_reference")
        axes[0, i].legend()
        
        # Plot u_reference
        axes[1, i].plot(time, u_ref_smooth[i, :], label="Smooth", linewidth=2)
        axes[1, i].plot(time, u_ref_linear[i, :], label="Linear", linestyle="--", linewidth=2)
        axes[1, i].set_title(f"u_reference (Dimension {i+1})")
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_ylabel("u_reference")
        axes[1, i].legend()

    plt.tight_layout()
    plt.show()

# # Call the test function
# test_generate_trajectory()
