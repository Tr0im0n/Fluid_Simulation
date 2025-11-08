
def square_kernel(distance: float, radius: float, inverse_volume: float) -> float:
    linear = max(0, radius - distance)
    return pow(linear, 2) * inverse_volume


def linear_kernel(distance: float, radius: float, inverse_volume: float) -> float:
    linear = max(0, radius - distance)
    return linear * inverse_volume


def n_power_kernel(distance: float, radius: float, inverse_volume: float, power: int) -> float:
    linear = max(0, radius - distance)
    return pow(linear, power) * inverse_volume
