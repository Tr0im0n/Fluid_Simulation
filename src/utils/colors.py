

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
