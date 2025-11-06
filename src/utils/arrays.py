
import numpy as np
from numpy.typing import NDArray


def create_coordinate_array(width: int, height: int) -> NDArray[np.int32]:
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers.")
    
    rows = np.arange(height)
    cols = np.arange(width)
    
    X_coords, Y_coords = np.meshgrid(cols, rows)
    
    return np.stack([X_coords, Y_coords], axis=2, dtype=np.int32)


def normalize_array(array: np.ndarray) -> np.ndarray:
    """ Return the array, but normalized. """
    min_val = array.min()
    max_val = array.max()
    if max_val == min_val:
        return np.full_like(array, 0.5)
    return (array - min_val) / (max_val - min_val)


def norm_ip(buffer2d: np.ndarray, buffer1d: np.ndarray):
    """Calcs the norm of buffer2d and puts into buffer1d"""
    if buffer1d.ndim != 1:
        raise Exception("buffer1d is not, in fact, 1d")
    if buffer2d.ndim != 2:
        raise Exception("buffer2d is not, in fact, 2d")
    if buffer1d.shape[0] != buffer2d.shape[0]:
        raise IndexError("Arrays aren't of same size")
    np.square(buffer2d, out=buffer2d)
    np.sum(buffer2d, axis=1, out=buffer1d)
    np.sqrt(buffer1d, out=buffer1d)
