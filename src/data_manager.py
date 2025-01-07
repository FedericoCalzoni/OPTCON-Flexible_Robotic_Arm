from datetime import datetime 
import numpy as np
import os
import re

def find_latest_versioned_file(base_name, directory="."):
    """
    Finds the latest versioned file based on version number in the filename.
    
    Parameters:
        base_name (str): Base name of the file.
        extension (str): File extension (e.g., ".npz").
        directory (str): Directory to search for files.

    Returns:
        str: Path to the latest versioned file, or None if no file is found.
    """

    extension = '.npz'
    # Regular expression to match filenames with version and timestamp
    pattern = re.compile(rf"{base_name}_v(\d+)_\d{{8}}_\d{{6}}{re.escape(extension)}")
    max_version = -1
    latest_file = None

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            version = int(match.group(1))  # Extract the version number
            if version > max_version:
                max_version = version
                latest_file = os.path.join(directory, filename)

    return latest_file


def get_next_filename(base_name, directory="."):
    """
    Generates the next incremental filename based on existing files in the directory.

    Parameters:
        base_name (str): Base name of the file.
        extension (str): File extension (e.g., ".npz").
        directory (str): Directory to search for existing files.

    Returns:
        int: The version of the last file found in the archive
    """
    
    extension = ".npz"
    pattern = rf"{re.escape(base_name)}_v(\d+)_.*{re.escape(extension)}"
    
    # Trova tutti i file nella directory che corrispondono al pattern
    versions = []
    for file_name in os.listdir(directory):
        match = re.match(pattern, file_name)
        if match:
            versions.append(int(match.group(1)))  # Estrai il numero di versione

    # Calcola la nuova versione
    new_version = max(versions, default=0) + 1  # Se nessun file trovato, inizia da 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file_name = f"{directory}\\{base_name}_v{new_version}_{timestamp}{extension}"
    return new_file_name




def save_optimal_trajectory(x_optimal, u_optimal):
    file = "Optimal_Trajectory"
    dir = "DataArchive"
    save_file(x_optimal, u_optimal, base_name=file, directory=dir)

def save_file(*args, base_name, directory):
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    archive_dir = os.path.join(base_dir, '..', directory)
    os.makedirs(archive_dir, exist_ok=True)
    new_file_name = get_next_filename(base_name, archive_dir)
    # Crea un dizionario con nomi dinamici per ogni argomento
    saved_data = {}

    for i, arg in enumerate(args):
        # Create dynamic keys
        var_name = f'arg{i}'
        saved_data[var_name] = arg

    np.savez(new_file_name, **saved_data)



