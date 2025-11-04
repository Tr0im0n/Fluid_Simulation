
import numpy as np
from numpy.typing import NDArray


RED   = np.array([255, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)
BLUE  = np.array([0, 0, 255], dtype=np.uint8)
BLACK  = np.array([0, 0, 0], dtype=np.uint8)

DARKGRAY = np.array([200, 200, 200], dtype=np.uint8)

FLOAT_BLUE = BLUE.astype(np.float32)
FLOAT_WHITE = WHITE.astype(np.float32)
FLOAT_RED = RED.astype(np.float32)


def colormap(normalized_val: float, color0, color1, color2) -> tuple[int, int, int]:
    """Maps a normalized density (0.0 to 1.0) to a Blue-White-Red color."""
    
    if normalized_val <= 0.5:
        t = normalized_val * 2.0
        r = int(color0[0] + t * (color1[0] - color0[0]))
        g = int(color0[1] + t * (color1[1] - color0[1]))
        b = int(color0[2] + t * (color1[2] - color0[2]))
    else:
        t = (normalized_val - 0.5) * 2.0
        r = int(color1[0] + t * (color2[0] - color1[0]))
        g = int(color1[1] + t * (color2[1] - color1[1]))
        b = int(color1[2] + t * (color2[2] - color1[2]))
    return (r, g, b)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamps a value between min_value and max_value."""
    return max(min_value, min(max_value, value))


def colormap_RWB(normalized_val: float) -> NDArray[np.uint8]:
    if normalized_val <= 0.5:
        t = normalized_val * 2.0
        color1 = BLUE
        color2 = WHITE
    else:
        t = (normalized_val - 0.5) * 2.0
        color1 = WHITE
        color2 = RED
    return (color1 + t * (color2 - color1)).astype(np.uint8)


def colormap_array_BWR(normalized_array: NDArray[np.float32]) -> NDArray[np.uint8]:
    rgb_array = np.zeros(normalized_array.shape + (3,), dtype=np.float32)
    
    mask_bw = normalized_array <= 0.5
    mask_wr = ~mask_bw # Bitwise NOT operator
    
    t_bw = normalized_array[mask_bw] * 2.0
    t_wr = (normalized_array[mask_wr] - 0.5) * 2.0
    
    rgb_array[mask_bw] = FLOAT_BLUE + t_bw[:, None] * (FLOAT_WHITE - FLOAT_BLUE)
    rgb_array[mask_wr] = FLOAT_WHITE + t_wr[:, None] * (FLOAT_RED - FLOAT_WHITE)

    return rgb_array.astype(np.uint8)
