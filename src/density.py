
import numpy as np

"""

This has taken heavy inspiration from the following video:
https://www.youtube.com/watch?v=rSKMYc1CQHE


"""


class DensityFluidSim:
    PI = np.pi
    MASS = 1.0
    RADIUS = 1.0

    def __init__(self, points: np.ndarray, radius: float | None = None) -> None:
        self.points = points
        self.n_points = len(points)
        self.radius = radius if radius is not None else self.RADIUS
        self.mass = 1.0
        self.target_density = 10.0
        self.pressure_force_multiplier = 0.5

        self.cached_densities = np.zeros(self.n_points, dtype=np.float32)

        # Precompute constants
        self.volume_of_inlfuence = (self.PI * pow(self.radius, 4)) / 6
        self.inverse_volume = 6.0 / (self.PI * pow(self.radius, 4))
        self.inverse_volume2 = -12.0 / (self.PI * pow(self.radius, 4))

    def calc_influence(self, distance: float) -> float:
        linear = max(0, self.radius - distance)
        return pow(linear, 2) * self.inverse_volume
    
    def calc_inlfuence_derivative(self, distance: float) -> float:
        linear = max(0, self.radius - distance)
        return linear * self.inverse_volume2

    def _density_at_point_loop(self, point: np.ndarray) -> float:
        total_density = 0.0
        for other_point in self.points:
            distance = float(np.linalg.norm(point - other_point))
            influence = self.calc_influence(distance)
            total_density += influence * self.mass
        return total_density

    def _density_at_point_vector(self, point: np.ndarray) -> float:
        diffs = self.points - point
        dists = np.linalg.norm(diffs, axis=1)
        linear = np.maximum(0.0, self.radius - dists)
        influences = np.power(linear, 2)
        return float(influences.sum() * self.mass * self.inverse_volume)

    def density_at_point(self, point: np.ndarray) -> float:
        return self._density_at_point_vector(point)

    def cache_densities(self) -> None:
        self.cached_densities.fill(0.0)
        for i, point in enumerate(self.points):
            self.cached_densities[i] = self.density_at_point(point)

    def calc_pressure_force_for_index(self, index: int) -> np.ndarray:
        point = self.points[index]
        density = self.cached_densities[index]

        total_force = np.zeros(2, dtype=np.float32)

        for j, other_point in enumerate(self.points):
            if j == index:
                continue
            diff = point - other_point
            distance = float(np.linalg.norm(diff))
            if distance > self.radius or distance == 0:
                continue
            direction = diff / distance
            influence_derivative = self.calc_inlfuence_derivative(distance)

            other_density = self.cached_densities[j]
            
            
            pressure_term = (density + other_density - 2 * self.target_density) * self.pressure_force_multiplier

            force_magnitude = -self.mass * pressure_term * influence_derivative
            total_force += force_magnitude * direction

        return total_force












