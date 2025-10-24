
import numpy as np
from itertools import chain

"""

This has taken heavy inspiration from the following video:
https://www.youtube.com/watch?v=rSKMYc1CQHE


"""


class DensityFluidSim:
    PI = np.pi
    DEFAULT_RADIUS = 50.0
    DEFAULTS_DICT = {
        "sim_width": 800,
        "sim_height": 600,
        "mass": 1.0,
        "target_density": 10.0,
        "pressure_force_multiplier": 0.5,
    }
    # instance attribute annotations so static type checkers know these exist
    sim_width: int
    sim_height: int
    mass: float
    smoothing_radius: float
    target_density: float
    pressure_force_multiplier: float

    def __init__(self,
                 points: np.ndarray,
                 # All DEFAULTS_DICT keys below:
                 sim_width: int | None = None,
                 sim_height: int | None = None,
                 radius: float | None = None,
                 mass: float | None = None,
                 target_density: float | None = None,
                 pressure_force_multiplier: float | None = None, 
                 ) -> None:
        
        init_args = locals()
        self._set_config_attributes(init_args)

        self.points = points
        self.n_points = len(points)

        self.radius = radius if radius is not None else self.DEFAULT_RADIUS
        
        self.spatial_partition_array = np.full(self.total_cells, fill_value=[], dtype=object)
        self.populate_spatial_partition()

        self.cached_densities = np.zeros(self.n_points, dtype=np.float32)
        self.cache_densities()


    def _set_config_attributes(self, init_args) -> None:
        for var_name, default_value in self.DEFAULTS_DICT.items():
            value = init_args.get(var_name)
            setattr(self, var_name, value if value is not None else default_value)

    @property
    def radius(self) -> float:
        return self._radius
    
    @radius.setter
    def radius(self, value: float) -> None: 
        self._radius = value
        # Update precomputed constants when radius changes
        self._calc_vals_dependent_on_radius

    def _calc_vals_dependent_on_radius(self) -> None:
        # Grid dimensions
        self.grid_width = int(self.sim_width / self.radius)
        self.grid_height = int(self.sim_height / self.radius)
        self.total_cells = self.grid_width * self.grid_height
        # Volumes and inverses
        self.volume_of_influence = (self.PI * pow(self.radius, 4)) / 6
        self.inverse_volume = 6.0 / (self.PI * pow(self.radius, 4))
        self.inverse_volume2 = -12.0 / (self.PI * pow(self.radius, 4))
        # 
        self.neighbors = [-self.grid_width -1, -self.grid_width, -self.grid_width +1,
                          -1,                 0,              1,
                          self.grid_width -1,  self.grid_width, self.grid_width +1]

    def partition_index_from_point(self, point: np.ndarray) -> int:
        cell_x = int(point[0] // self.radius)
        cell_y = int(point[1] // self.radius)
        return cell_y * self.grid_width + cell_x

    def cell_coords_from_index(self, index: int) -> np.ndarray:
        cell_y = index // self.grid_width
        cell_x = index % self.grid_width
        return np.array([cell_x, cell_y], dtype=np.int32)

    def populate_spatial_partition(self) -> None:
        for i, point in enumerate(self.points):
            cell_index = self.partition_index_from_point(point)
            self.spatial_partition_array[cell_index].append(i)

    def calc_influence(self, distance: float) -> float:
        linear = max(0, self.radius - distance)
        return pow(linear, 2) * self.inverse_volume
    
    def calc_influence_derivative(self, distance: float) -> float:
        linear = max(0, self.radius - distance)
        return linear * self.inverse_volume2

    def neighboring_cell_indices(self, point: np.ndarray) -> list[int]:
        current_index = self.partition_index_from_point(point)
        neighbor_cell_indices = [
            current_index + off
            for off in self.neighbors
            if 0 <= (current_index + off) < self.total_cells
        ]
        return neighbor_cell_indices

    def neighboring_particles(self, point: np.ndarray) -> np.ndarray:
        neighbor_cell_indices = self.neighboring_cell_indices(point)
        list_of_lists_of_particles = [self.spatial_partition_array[i] for i in neighbor_cell_indices]
        return np.concatenate(list_of_lists_of_particles, axis=0)

    def _density_at_point_spatial_partition(self, point: np.ndarray) -> float:
        particles_to_check = self.neighboring_particles(point)

        if 0 == particles_to_check.size:
            return 0.0

        diffs = particles_to_check - point
        dists = np.linalg.norm(diffs, axis=1)
        linear = np.maximum(0.0, self.radius - dists)
        influences = linear * linear

        return influences.sum() * self.mass * self.inverse_volume

    def density_at_point(self, point: np.ndarray) -> float:
        return self._density_at_point_spatial_partition(point)

    def cache_densities(self) -> None:
        """Could be optimized with Numba"""
        self.cached_densities.fill(0.0)
        for i, point in enumerate(self.points):
            self.cached_densities[i] = self.density_at_point(point)

#########################################################################################################################

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
            influence_derivative = self.calc_influence_derivative(distance)

            other_density = self.cached_densities[j]
            

            pressure_term = (density + other_density - 2 * self.target_density) * self.pressure_force_multiplier

            force_magnitude = -self.mass * pressure_term * influence_derivative
            total_force += force_magnitude * direction

        return total_force

#########################################################################################################################
    # Depreciated functions
#########################################################################################################################

    def density_contribution_from_cell(self, point: np.ndarray, cell_index: int) -> float:
        points_in_cell = self.spatial_partition_array[cell_index]
        diffs = points_in_cell - point
        dists = np.linalg.norm(diffs, axis=1)
        linear = np.maximum(0.0, self.radius - dists)
        influences = np.power(linear, 2)
        return float((influences.sum() * self.mass * self.inverse_volume))

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








