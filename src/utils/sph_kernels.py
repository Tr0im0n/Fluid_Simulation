
def square_kernel(distance: float, radius: float, inverse_volume: float) -> float:
    linear = max(0, radius - distance)
    return pow(linear, 2) * inverse_volume


def square_kernel_derivative(distance: float, radius: float, inverse_volume: float) -> float:
    linear = max(0, radius - distance)
    return linear * inverse_volume
