# OPTCON-Flexible_Robotic_Arm

## Setup
 
Install the environment:
 
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
python src/main.py

## Project 
Most relevant files:
    main.py:
    - Used to call functions and to achieve tasks given by the assignment
    - Setting the array "task_to_run" makes you able to run the task set. 
        Setting task_to_run = [1, 2, 3, 4] runs every task at once.
    
    parameters.py:
    - Used to define some configurations parameters, as a console.
    - Several variables can be set to true or false to trigger plots and/or 
        trajectory load. (Have a look at "data_manager.py")
    
Utilities:
    data_manager.py:
    - Used to store trajectories, so that the computational time is lowered.
    - Save functions put a version and a timestamp in the file name, and store
        it in the DataArchive directory.
    - Load functions look for a proper trajectory, with given version.

    symbolic_dyn.py:
    - Used to derive dynamics gradients and hessians. Only gradients are used.

    visualizer.py:
    - Used to animate the double pendulum.

Tasks:
    armijo.py
    - This files stores the function to implement the armijo step size selection 
        rule, and its relative plots. 

    costTask1-2-3-4.py:
    - These files are the exact same code but for a string.
    - Each file imports a different cost matrices set to use for each task.
    - Here is stored the cost function, cost gradient and hessian.

    dynamics.py:
    - This file is the numeric equivalent of symbolic_dyn.py.
    - It stores every phisical parameters.
    - It is called to compute dynamics and jacobians.

    LQR.py:
    - Runs a Linear Quadratic Regulator around an optimal
        trajectory computed in task 2.

    mpc.py:
    - Runs a Model Predictive Controller around an optimal
        trajectory computed in task 2.

    newton_method.py:
    - Root finding routine used to clearly define proper equilibria

    newton_for_optcon.py:
    - Runs the Newton Method for Optimal Control in a closed loop
        update form. 

    reference_trajectory.py:
    - Used to generate a proper reference trajectory to be adapted to system
        dynamics by Newton Method for Optimal Control.
