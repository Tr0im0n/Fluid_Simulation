
import numpy as np


def create_coordinate_array(width: int, height: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers.")
    
    rows = np.arange(height)
    cols = np.arange(width)
    
    X_coords, Y_coords = np.meshgrid(cols, rows)
    
    return np.stack([Y_coords, X_coords], axis=2)


def normalize_array(array: np.ndarray) -> np.ndarray:
    """ Return the array, but normalized. """
    min_val = array.min()
    max_val = array.max()
    if max_val == min_val:
        return np.full_like(array, 0.5)
    return (array - min_val) / (max_val - min_val)
