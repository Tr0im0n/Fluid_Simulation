
import numpy as np
from logic.spatial_partition_list import SpatialPartitionList
from src.utils.instance import _set_config_attributes
from src.utils.sph_kernels import square_kernel, square_kernel_derivative

"""

This has taken heavy inspiration from the following video:
https://www.youtube.com/watch?v=rSKMYc1CQHE


"""


class DensityFluidSim:
    PI = np.pi
    DEFAULT_RADIUS = 50.0
    DIMENSIONS = 2
    N_NEIGHBORS = 9
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
    target_density: float
    pressure_force_multiplier: float
    spatial_partition_list: SpatialPartitionList

    def __init__(
            self,
            particles: np.ndarray,
            # All DEFAULTS_DICT keys below:
            sim_width: int | None = None,
            sim_height: int | None = None,
            radius: float | None = None,
            mass: float | None = None,
            target_density: float | None = None,
            pressure_force_multiplier: float | None = None, 
        ) -> None:
        
        init_args = locals()
        _set_config_attributes(self, init_args, self.DEFAULTS_DICT)

        self._radius = radius if radius is not None else self.DEFAULT_RADIUS
        self._particles = particles
        self.spatial_partition_list = SpatialPartitionList(self._particles, self._radius, self.sim_width, self.sim_height)
        
        self._calc_vals_dependent_on_radius()
        self._on_particles_change(True, True)

#################################################################################################
# Properties 
#################################################################################################

    @property
    def radius(self) -> float:
        return self._radius
    
    @radius.setter
    def radius(self, value: float) -> None: 
        self._radius = value
        self._calc_vals_dependent_on_radius()
        self.new_spatial_partition()

    def _calc_vals_dependent_on_radius(self) -> None:
        """ Calculate volumes and inverses. """
        self.volume_of_influence = (self.PI * pow(self.radius, 4)) / 6
        self.inverse_volume = 6.0 / (self.PI * pow(self.radius, 4))
        self.inverse_volume2 = -12.0 / (self.PI * pow(self.radius, 4))

# Particle property ##############################################################################

    @property
    def particles(self) -> np.ndarray:
        return self._particles
    
    @particles.setter
    def particles(self, value: np.ndarray) -> None: 
        self._particles = value
        self._on_particles_change(change_amount=True)

    def set_particles(self, value: np.ndarray, change_amount: bool = True, new_spatial_partition: bool = False) -> None:
        self._particles = value
        self._on_particles_change(change_amount, new_spatial_partition)
        
    def _on_particles_change(self, change_amount: bool = False, new_spatial_partition: bool = False) -> None:
        if change_amount:
            self.n_particles = self._particles.shape[0]
            
        self.populate_spatial_partition(create_new_list=new_spatial_partition)
        self.cache_densities(create_new_array=change_amount)

#################################################################################################
# Methods
#################################################################################################

    def new_spatial_partition(self):
        """ Make new spatial partition, most likely due to change of influence radius. """ 
        self.spatial_partition_list = SpatialPartitionList(
            self._particles, self._radius, self.sim_width, self.sim_height)

    def _calc_density_at_point_spatial_partition(self, point: np.ndarray) -> float:
        """ Calculate the density at the given points. 
        Optimized by only looking at particles in neighboring partitions. """
        neighbor_indices = self.get_neighboring_particle_indices(point)
        neighbor_particles = self._particles[neighbor_indices]
        # if 0 == neighbor_particles.size:
        #     return 0.0
        diffs = neighbor_particles - point
        dists = np.linalg.norm(diffs, axis=1)
        linear = np.maximum(0.0, self.radius - dists)
        influences = linear * linear

        return influences.sum() * self.mass * self.inverse_volume

    def calc_density_at_point(self, point: np.ndarray) -> float:
        """ Calculate the density at the given points. 
        Pick one of the private methods to use. """
        return self._calc_density_at_point_spatial_partition(point)

    def cache_densities(self, create_new_array: bool = False) -> None:
        """Could be optimized with Numba"""
        if create_new_array:
            self.cached_densities = np.zeros(self.n_particles, dtype=np.float32)

        self.cached_densities.fill(0.0)
        for i, point in enumerate(self._particles):
            self.cached_densities[i] = self.calc_density_at_point(point)

    def get_normalized_densities(self) -> np.ndarray:
        """ Return the array of densities, but normalized. """
        rho_min = self.cached_densities.min()
        rho_max = self.cached_densities.max()
        if rho_max == rho_min:
            normalized_densities = np.full_like(self.cached_densities, 0.5)
        else:
            normalized_densities = (self.cached_densities - rho_min) / (rho_max - rho_min)
        return normalized_densities

    def generate_random_particles(self, num_points:int, seed: int = 42) -> np.ndarray:
        """ I mean read the method name. """
        np.random.seed(seed)
        max_width, max_height = self.sim_width, self.sim_height
        particles = np.random.uniform(
            low=[0, 0], 
            high=[max_width, max_height], 
            size=(num_points, 2)
        )
        return particles

#########################################################################################################################

    def calc_pressure_force_for_index(self, index: int) -> np.ndarray:
        point = self._particles[index]
        density = self.cached_densities[index]

        total_force = np.zeros(2, dtype=np.float32)

        for j, other_point in enumerate(self._particles):
            if j == index:
                continue
            diff = point - other_point
            distance = float(np.linalg.norm(diff))
            if distance > self.radius or distance == 0:
                continue
            direction = diff / distance
            influence_derivative = square_kernel_derivative(distance, self._radius, self.inverse_volume2)

            other_density = self.cached_densities[j]
            

            pressure_term = (density + other_density - 2 * self.target_density) * self.pressure_force_multiplier

            force_magnitude = -self.mass * pressure_term * influence_derivative
            total_force += force_magnitude * direction

        return total_force


    @classmethod
    def particle_grid(cls, amount: int) -> np.ndarray:
        """This seems quite shit. """
        side_length = int(np.ceil(np.sqrt(amount)))
        spacing = cls.DEFAULT_RADIUS * 0.5
        points = []
        for i in range(amount):
            x = (i % side_length) * spacing + spacing
            y = (i // side_length) * spacing + spacing
            points.append([x, y])
        return np.array(points, dtype=np.float32)

#########################################################################################################################
    # Depreciated functions
#########################################################################################################################

def main():
    particle_grid = DensityFluidSim.particle_grid(10)
    DFS = DensityFluidSim(particle_grid)
    random_particles = DFS.generate_random_particles(1000)
    DFS.set_particles(random_particles)


if __name__ == "__main__":
    main()

