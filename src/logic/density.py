
import math
import numpy as np
from src.logic.spatial_partition_list import SpatialPartitionList
from src.utils.arrays import create_coordinate_array
from src.utils.instance import _set_config_attributes
from src.utils.sph_kernels import square_kernel, square_kernel_derivative

"""

This has taken heavy inspiration from the following video:
https://www.youtube.com/watch?v=rSKMYc1CQHE


"""


class DensityFluidSim:
    PI = np.pi
    DIMENSIONS: int = 2
    N_NEIGHBORS: int = 9

    def __init__(self,
            particles: np.ndarray | None = None,
            sim_width: int = 800,
            sim_height: int = 600,
            radius: float = 50.0,
            mass: float = 1.0,
            target_density: float = 10.0,
            pressure_force_multiplier: float = 0.5, 
        ) -> None:
        
        self.sim_width = sim_width
        self.sim_height = sim_height
        self._radius = radius
        self.mass = mass
        self.target_density = target_density
        self.pressure_force_multiplier = pressure_force_multiplier
        
        self._particles = particles if particles is not None else self.generate_random_particles()
        self.spatial_partition_list = SpatialPartitionList(self._particles, self._radius, self.sim_width, self.sim_height)
        
        self._calc_vals_dependent_on_radius()
        
        self.cache_densities(True)
        
        self.density_image = self.create_density_image()

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

    def set_particles(self, value: np.ndarray, change_amount: bool = True) -> None:
        self._particles = value
        self._on_particles_change(change_amount)
        
    def _on_particles_change(self, change_amount: bool = False) -> None:
        self.spatial_partition_list.populate(self.particles)
        self.cache_densities(create_new_array=change_amount)

#################################################################################################
# Methods
#################################################################################################

    def new_spatial_partition(self):
        """ Make new spatial partition, most likely due to change of influence radius. """ 
        self.spatial_partition_list = SpatialPartitionList(
            self._particles, self._radius, self.sim_width, self.sim_height)

    def calc_density_at_point(self, point: np.ndarray) -> float:
        """ Calculate the density at the given points. 
        Optimized by only looking at particles in neighboring partitions. """
        neighbor_indices = self.spatial_partition_list.get_neighboring_particle_indices(point)
        neighbor_particles = self._particles[neighbor_indices]
        # if 0 == neighbor_particles.size:
        #     return 0.0
        diffs = neighbor_particles - point
        dists = np.linalg.norm(diffs, axis=1)
        linear = np.maximum(0.0, self.radius - dists)
        influences = linear * linear

        return influences.sum() * self.mass * self.inverse_volume

    def cache_densities(self, create_new_array: bool = False) -> None:
        """Could be optimized with Numba"""
        if create_new_array:
            n_particles = self._particles.shape[0]
            self.cached_densities = np.zeros(n_particles, dtype=np.float32)

        self.cached_densities.fill(0.0)
        for i, point in enumerate(self._particles):
            self.cached_densities[i] = self.calc_density_at_point(point)

    def generate_random_particles(self, num_points:int = 1000, seed: int = 42) -> np.ndarray:
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

    def create_density_image(self) -> np.ndarray:
        image = np.zeros((self.sim_height, self.sim_width))
        # get the influence image of one particle
        # add that to the image
        # Literal edge cases where the influence image is not 9 cells
        # for i, particles in self.spatial_partition_list:
        #     neighbor_cells = self.spatial_partition_list.NEIGHBORING_CELLS_ARRAY[i]
        ceil_radius = math.ceil(self._radius)
        coordinate_array = create_coordinate_array(self.sim_width, self.sim_height)
        for particle in self._particles:
            min_x = max(0, particle[0] - ceil_radius)
            max_x = min(self.sim_width, particle[0] + ceil_radius)
            min_y = max(0, particle[1] - ceil_radius)
            max_y = min(self.sim_height, particle[1] + ceil_radius)
            
            influence_area = coordinate_array[min_x:max_x, min_y:max_y]
            diffs = influence_area - particle
            dists = np.linalg.norm(diffs, axis=2)
            linear = np.maximum(0.0, self.radius - dists)
            influences = pow(linear, 2) * self.mass * self.inverse_volume
 
            image[min_x:max_x, min_y:max_y] += influences
        return image
            

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

#########################################################################################################################
    # Depreciated functions
#########################################################################################################################

def main():
    DFS = DensityFluidSim()
    random_particles = DFS.generate_random_particles(1000)
    DFS.set_particles(random_particles)


if __name__ == "__main__":
    main()

