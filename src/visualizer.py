import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import parameters as pm

def animate_double_pendulum(x_history, title="Double Pendulum Animation", speed=1):
    """
    Create an animation of a double pendulum.

    Parameters:
        x_history (numpy.ndarray): Array containing the simulation data with shape (T, 4).
        speed (int): Number to change the speed of the animation.
    """

    dt = pm.dt
    L1 = pm.L1
    L2 = pm.L2
        
    num_frames = x_history.shape[0]
    frame_skip = int(num_frames * dt/speed)
    frame_skip += 1
    num_steps = num_frames // frame_skip
    
    positions = np.zeros((num_frames, 4))

    # Calculate positions of the pendulum
    positions[:, 0] = L1 * np.sin(x_history[:, 2])  # x1
    positions[:, 1] = -L1 * np.cos(x_history[:, 2])  # y1
    positions[:, 2] = positions[:, 0] + L2 * np.sin(x_history[:, 2]+x_history[:, 3])  # x2
    positions[:, 3] = positions[:, 1] - L2 * np.cos(x_history[:, 2]+x_history[:, 3])  # y2

    # Reduce data for animation by skipping frames
    positions = positions[::frame_skip]
    
    # Constants for the animation
    trace_length1 = 150  # Maximum length of the first hinge trace
    trace_length2 = 200  # Maximum length of the pendulum trace
    alpha_min, alpha_max = 0.1, 1  # Fading alpha range

    # Precompute alphas for fading traces
    alpha_values1 = np.linspace(alpha_min, alpha_max, trace_length1)
    alpha_values2 = np.linspace(alpha_min, alpha_max, trace_length2)

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3.1, 3)
    ax.set_aspect('equal')

    # Plot objects
    line, = ax.plot([], [], 'o-', lw=2)  # Pendulum rods
    trace1 = ax.scatter([], [], c=[], s=2, cmap='YlGn', vmin=0, vmax=1)  # Trace of the first hinge
    trace2 = ax.scatter([], [], c=[], s=2, cmap='Reds', vmin=0, vmax=1)  # Trace of the pendulum


    # Lists to store trace points
    trace_x1, trace_y1 = [], [] 
    trace_x2, trace_y2 = [], []

    def init():
        """Initialize the animation by clearing the data."""
        line.set_data([], [])
        trace1.set_offsets(np.empty((0, 2)))
        trace2.set_offsets(np.empty((0, 2)))
        trace_x1.clear()
        trace_y1.clear()
        trace_x2.clear()
        trace_y2.clear()
        return line, trace1, trace2

    def update(frame):
        """Update the animation for the given frame."""
        x1, y1, x2, y2 = positions[frame]

        # Set pendulum line positions
        line.set_data([0, x1, x2], [0, y1, y2])

        # Append the trace points
        trace_x1.append(x1)
        trace_y1.append(y1)
        trace_x2.append(x2)
        trace_y2.append(y2)

        # Limit trace size for better performance
        if len(trace_x1) >= trace_length1:
            trace_x1.pop(0)
            trace_y1.pop(0)
        if len(trace_x2) >= trace_length2:
            trace_x2.pop(0)
            trace_y2.pop(0)

        # Update scatter plots
        trace1.set_offsets(np.c_[trace_x1, trace_y1])
        trace1.set_array(alpha_values1[-len(trace_x1):])

        trace2.set_offsets(np.c_[trace_x2, trace_y2])
        trace2.set_array(alpha_values2[-len(trace_x2):])

        return line, trace1, trace2

    # Create the animation
    animation = FuncAnimation(
        fig, update,
        frames=num_steps,
        init_func=init,
        blit=True,
        interval=0
    )

    plt.show()
