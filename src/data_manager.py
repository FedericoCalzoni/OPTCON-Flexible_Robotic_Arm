from datetime import datetime 
import numpy as np
import os
import re

#################################
##        Save Functions       ##
#################################

def save_optimal_trajectory(x_optimal, u_optimal):
    """
    Save the optimal trajectory data on a specified version of the file "Optimal_Trajectories".

    Args:
        x_optimal (NumPy.array): The optimal trajectory of the states.
        u_optimal (NumPy.array): The optimal trajectory of the inputs.
    Returns:
        None

    Example:
        >>> save_optimal_trajectory(x_opt, u_opt)
    """
    file = "Optimal_Trajectory"
    dir = "DataArchive"
    _save_file(x_optimal, u_optimal, base_name=file, directory=dir)


def save_mpc_trajectory(x_mpc, u_mpc):
    """
    Save the optimal trajectory data on a specified version of the file "Optimal_Trajectories".

    Args:
        x_optimal (NumPy.array): The optimal trajectory of the states.
        u_optimal (NumPy.array): The optimal trajectory of the inputs.
    Returns:
        None

    Example:
        >>> save_optimal_trajectory(x_opt, u_opt)
    """
    file = "MPC_Trajectory"
    dir = "DataArchive"
    _save_file(x_mpc, u_mpc, base_name=file, directory=dir)

def save_lqr_trajectory(x_LQR, u_LQR):
    """
    Save the optimal trajectory data on a specified version of the file "Optimal_Trajectories".

    Args:
        x_optimal (NumPy.array): The optimal trajectory of the states.
        u_optimal (NumPy.array): The optimal trajectory of the inputs.
    Returns:
        None

    Example:
        >>> save_optimal_trajectory(x_opt, u_opt)
    """
    file = "LQR_Trajectory"
    dir = "DataArchive"
    _save_file(x_LQR, u_LQR, base_name=file, directory=dir)

def _save_file(*args, base_name, directory):
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    archive_dir = os.path.join(base_dir, '..', directory)
    os.makedirs(archive_dir, exist_ok=True)
    new_file_name = _get_next_filename(base_name, archive_dir)
    # Create a dictionary with dynamic names for each argument
    saved_data = {}

    for i, arg in enumerate(args):
        # Create dynamic keys
        var_name = f'arg{i}'
        saved_data[var_name] = arg

    np.savez(new_file_name, **saved_data)
    print(f"\nFile Saved:\t{base_name}\t in\t {directory}")

def _get_next_filename(base_name, directory="."):
    """
    Generates the next incremental filename based on existing files in the directory.

    Parameters:
        base_name (str): Base name of the file.
        extension (str): File extension (e.g., ".npz").
        directory (str): Directory to search for existing files.

    Returns:
        str: The properly formatted file name to be written
    """
    extension = ".npz"
    pattern = rf"{re.escape(base_name)}_v(\d+)_.*{re.escape(extension)}"
    
    # Find all files in the directory that match the pattern
    versions = []
    for file_name in os.listdir(directory):
        match = re.match(pattern, file_name)
        if match:
            versions.append(int(match.group(1)))  # Extract the version number

    # Calculate the new version
    new_version = max(versions, default=0) + 1  # Start from 1 if no file found

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file_name = os.path.join(directory, f"{base_name}_v{new_version}_{timestamp}{extension}")
    return new_file_name

#################################
##        Load Functions       ##
#################################

def load_optimal_trajectory(version='latest'):
    """
    Loads the optimal trajectory data from a specified version of the file.

    This function retrieves the optimal state (`x_optimal`) and control (`u_optimal`) trajectories
    from a file stored in a predefined directory. The version of the file can be specified, and the
    default is the latest version.

    Args:
        version (str, optional): The version of the file to load. Defaults to 'latest'.

    Returns:
        tuple: A tuple containing:
            - x_optimal (any): The optimal state trajectory loaded from the file.
            - u_optimal (any): The optimal control trajectory loaded from the file.

    Raises:
        FileNotFoundError: If the file specified by the parameters is not found.
        KeyError: If the expected keys ('arg0', 'arg1') are not present in the loaded file.
        Exception: For other unforeseen errors during file loading or processing.

    Example:
        >>> x_opt, u_opt = load_optimal_trajectory(version='3')
    """
    file = "Optimal_Trajectory"
    dir = "DataArchive"

    trajectories = _load_file(file, dir, version)
    
    for arg_name in trajectories:
        match arg_name:
            case 'arg0':
                x_optimal = trajectories[arg_name]
            case 'arg1':
                u_optimal = trajectories[arg_name]
    return x_optimal, u_optimal

def load_mpc_trajectory(version = 'latest'):
    """
    Loads the optimal trajectory data from a specified version of the file.

    This function retrieves the optimal state (`x_optimal`) and control (`u_optimal`) trajectories
    from a file stored in a predefined directory. The version of the file can be specified, and the
    default is the latest version.

    Args:
        version (str, optional): The version of the file to load. Defaults to 'latest'.

    Returns:
        tuple: A tuple containing:
            - x_optimal (any): The optimal state trajectory loaded from the file.
            - u_optimal (any): The optimal control trajectory loaded from the file.

    Raises:
        FileNotFoundError: If the file specified by the parameters is not found.
        KeyError: If the expected keys ('arg0', 'arg1') are not present in the loaded file.
        Exception: For other unforeseen errors during file loading or processing.

    Example:
        >>> x_opt, u_opt = load_optimal_trajectory(version='3')
    """
    file = "MPC_Trajectory"
    dir = "DataArchive"

    trajectories = _load_file(file, dir, version)
    
    for arg_name in trajectories:
        match arg_name:
            case 'arg0':
                x_mpc = trajectories[arg_name]
            case 'arg1':
                u_mpc = trajectories[arg_name]
    return x_mpc, u_mpc

def load_lqr_trajectory(version = 'latest'):
    """
    Loads the optimal trajectory data from a specified version of the file.

    This function retrieves the optimal state (`x_optimal`) and control (`u_optimal`) trajectories
    from a file stored in a predefined directory. The version of the file can be specified, and the
    default is the latest version.

    Args:
        version (str, optional): The version of the file to load. Defaults to 'latest'.

    Returns:
        tuple: A tuple containing:
            - x_optimal (any): The optimal state trajectory loaded from the file.
            - u_optimal (any): The optimal control trajectory loaded from the file.

    Raises:
        FileNotFoundError: If the file specified by the parameters is not found.
        KeyError: If the expected keys ('arg0', 'arg1') are not present in the loaded file.
        Exception: For other unforeseen errors during file loading or processing.

    Example:
        >>> x_opt, u_opt = load_optimal_trajectory(version='3')
    """
    file = "LQR_Trajectory"
    dir = "DataArchive"

    trajectories = _load_file(file, dir, version)
    
    for arg_name in trajectories:
        match arg_name:
            case 'arg0':
                x_lqr = trajectories[arg_name]
            case 'arg1':
                u_lqr = trajectories[arg_name]
    return x_lqr, u_lqr

def _load_file(base_name, directory, version = 'latest'):
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    archive_dir = os.path.join(base_dir, '..', directory)
    os.makedirs(archive_dir, exist_ok=True)
    file_name_to_be_loaded  = _get_filename_to_load(base_name, archive_dir, version)
    vars = np.load(file_name_to_be_loaded)

    return vars

def _get_filename_to_load(base_name, directory, version):
    extension = ".npz"
    file_name_to_load = None
    pattern = rf"{re.escape(base_name)}_v(\d+)_.*{re.escape(extension)}"
    
    # Find all files in the directory that match the pattern
    versions = []
    for file_name in os.listdir(directory):
        match = re.match(pattern, file_name)
        if match:
            versions.append(int(match.group(1)))  # Extract the version number

    try:
        if version == 'latest':
            found_version = max(versions, default=0)
            if found_version != 0:
                print(f"Loading {base_name}, version {found_version}, from dir {directory}")
            else:
                raise ValueError(f"\nThere are no available version of {base_name} in {directory}.")

        elif version != 'latest':
            if int(version) in versions:
                found_version = versions[int(version)-1]
                print(f"Loading {base_name}, version {found_version}, from dir {directory}")
            else:
                raise ValueError(f"\nNo valid version '{version}' of {base_name} found in {directory}.")        
    except ValueError as e:
        print(f"Loading File Error: {e}") 
    else:
        second_pattern = rf"{re.escape(base_name)}_v{found_version}_.*{re.escape(extension)}"
        for file_name in os.listdir(directory):
            if re.match(second_pattern, file_name):
                file_name_to_load = os.path.join(directory, file_name)
                break

    return file_name_to_load
