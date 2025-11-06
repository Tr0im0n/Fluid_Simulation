
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
        
        self._particles = particles if particles is not None else self.generate_random_particles(1000, self.sim_width, self.sim_height)
        self.spatial_partition_list = SpatialPartitionList(self._particles, self._radius, self.sim_width, self.sim_height)
        
        self._calc_vals_dependent_on_radius()
        
        self.update_densities(True)
        
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
        self.update_densities(create_new_array=change_amount)

#################################################################################################
# Instance Methods
#################################################################################################

    def new_spatial_partition(self):
        """ Make new spatial partition, most likely due to change of influence radius. """ 
        self.spatial_partition_list = SpatialPartitionList(
            self._particles, self._radius, self.sim_width, self.sim_height)
        
    def update_densities(self, create_new_array: bool = False) -> None:
        """ Check if we need to make a new array. Needed if we add more particles.
            Call function to change the density in place"""
        if create_new_array:
            n_particles = self._particles.shape[0]
            self.densities_of_particles = np.zeros(n_particles, dtype=np.float32)
        else:
            self.densities_of_particles.fill(0.0)
            
        self.calc_densities_1loop(self.densities_of_particles, self._particles, self.spatial_partition_list,
                            self._radius, self.inverse_volume, self.mass)

    def create_density_image(self) -> np.ndarray:
        """
        get the influence image of one particle
        add that to the image
        """
        return self.calc_density_image(self._particles, self.sim_width, self.sim_height, 
                                       self._radius, self.inverse_volume, self.mass)

#################################################################################################
# Static Methods
#################################################################################################

    @staticmethod
    def generate_random_particles(num_points:int = 1000, max_width: float = 100., 
                                  max_height: float = 100., seed: int = 42) -> np.ndarray:
        """ I mean read the method name. """
        np.random.seed(seed)
        # max_width, max_height = self.sim_width, self.sim_height
        particles = np.random.uniform(
            low=[0, 0], 
            high=[max_width, max_height], 
            size=(num_points, 2)
        )
        return particles

    @staticmethod
    def calc_densities_1loop(cached_densities: np.ndarray, particles: np.ndarray, spl: SpatialPartitionList,
                       radius: float = 40., inverse_volume: float = 1., mass: float = 1.) -> None:
        """
        In place for speed.
        Could be optimized with Numba.
        """
        max_size = max(arr.size for arr in spl.cell_to_particle_neighbors)
        buffer1d = np.empty((max_size,), dtype=np.float32)
        buffer2d = np.empty((max_size, 2), dtype=np.float32)
        
        for i, point in enumerate(particles):
            cell_index = spl.get_partition_index_from_pos(point)
            neighbor_indices = spl.cell_to_particle_neighbors[cell_index]
            
            current_length = neighbor_indices.size
            b1d = buffer1d[:current_length]
            b2d = buffer2d[:current_length]
            
            np.take(particles, neighbor_indices, axis=0, out=b2d) # neighbor_particles
            # if 0 == neighbor_particles.size:
            #     return 0.0    
            b2d -= point # diffs
            
            # buffer1d[:current_length] = np.linalg.norm(buffer2d[:current_length], axis=1) # dists
            np.square(b2d, out=b2d)
            np.sum(b2d, axis=1, out=b1d)
            np.sqrt(b1d, out=b1d)
            
            np.maximum(0.0, radius - b1d, out=b1d) # linear
            b1d *= b1d # influence

            cached_densities[i] = b1d.sum() * mass * inverse_volume
            
    @staticmethod
    def calc_densities_vectorized(cached_densities: np.ndarray, particles: np.ndarray, spl: SpatialPartitionList,
                                  radius: float = 40., inverse_volume: float = 1., mass: float = 1.) -> None:
        """
        In place for speed.
        Could be optimized with Numba.
        """
        max_size = max(arr.size for arr in spl.cell_to_particle_neighbors)
        
        
        
        
        buffer1d = np.empty((max_size,), dtype=np.float32)
        buffer2d = np.empty((max_size, 2), dtype=np.float32)
        
        for i, point in enumerate(particles):
            cell_index = spl.get_partition_index_from_pos(point)
            neighbor_indices = spl.cell_to_particle_neighbors[cell_index]
            
            current_length = neighbor_indices.size
            b1d = buffer1d[:current_length]
            b2d = buffer2d[:current_length]
            
            np.take(particles, neighbor_indices, axis=0, out=b2d) # neighbor_particles
            # if 0 == neighbor_particles.size:
            #     return 0.0    
            b2d -= point # diffs
            
            # buffer1d[:current_length] = np.linalg.norm(buffer2d[:current_length], axis=1) # dists
            np.square(b2d, out=b2d)
            np.sum(b2d, axis=1, out=b1d)
            np.sqrt(b1d, out=b1d)
            
            np.maximum(0.0, radius - b1d, out=b1d) # linear
            b1d *= b1d # influence

            cached_densities[i] = b1d.sum() * mass * inverse_volume
    
    @staticmethod
    def calc_density_image(particles: np.ndarray, sim_width: int, sim_height: int, 
                             radius: float, inverse_volume: float, mass: float|int = 1) -> np.ndarray:
        """
        get the influence image of one particle
        add that to the image
        """
        image = np.zeros((sim_height, sim_width), dtype=np.float32)
        coordinate_array = create_coordinate_array(sim_width, sim_height)
        
        for particle in particles:
            min_x = max(0, math.ceil(particle[0] - radius))
            max_x = min(sim_width, math.floor(particle[0] + radius))
            min_y = max(0, math.ceil(particle[1] - radius))
            max_y = min(sim_height, math.floor(particle[1] + radius))
            
            influence_area = coordinate_array[min_y:max_y, min_x:max_x]
            diffs = influence_area - particle
            dists = np.linalg.norm(diffs, axis=2)
            linear = np.maximum(0.0, radius - dists)
            influences = pow(linear, 2) * mass * inverse_volume
 
            image[min_y:max_y, min_x:max_x] += influences
        return image
    
#########################################################################################################################

    def calc_density_gradient_at_point(self):
        print(2)

    def calc_density_gradient(self, particles: np.ndarray, spl: SpatialPartitionList, radius: float, 
                              cached_densities: np.ndarray, mass: float, inverse_volume: float):
        
        for i, point in enumerate(particles):
            cell_index = spl.get_partition_index_from_pos(point)
            neighbor_indices = spl.cell_to_particle_neighbors[cell_index]
            neighbor_particles = particles[neighbor_indices]
            # if 0 == neighbor_particles.size:
            #     return 0.0
            diffs = neighbor_particles - point
            dists = np.linalg.norm(diffs, axis=1)
            linear = np.maximum(0.0, radius - dists)
            influences = linear * linear

            cached_densities[i] = influences.sum() * mass * inverse_volume

    def calc_pressure_force_for_index(self, index: int) -> np.ndarray:
        point = self._particles[index]
        density = self.densities_of_particles[index]

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

            other_density = self.densities_of_particles[j]
            

            pressure_term = (density + other_density - 2 * self.target_density) * self.pressure_force_multiplier

            force_magnitude = -self.mass * pressure_term * influence_derivative
            total_force += force_magnitude * direction

        return total_force

#########################################################################################################################
    # Depreciated functions
#########################################################################################################################

def main():
    DFS = DensityFluidSim()


if __name__ == "__main__":
    main()

