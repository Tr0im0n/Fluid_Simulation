
import numpy as np
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
    smoothing_radius: float
    target_density: float
    pressure_force_multiplier: float
    total_cells: int

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

        self.radius = radius if radius is not None else self.DEFAULT_RADIUS

        self.set_particles(particles, True, True)

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
        self.neighbor_offsets = np.array([
            -self.grid_width -1, -self.grid_width, -self.grid_width +1,
            -1,                 0,              1,
            self.grid_width -1,  self.grid_width, self.grid_width +1
        ])
        self.create_neighboring_cells_array()

    @property
    def particles(self) -> np.ndarray:
        return self._particles
    
    @particles.setter
    def particles(self, value: np.ndarray) -> None: 
        self._particles = value
        self._on_particles_change(change_amount=True)

    def set_particles(self, value: np.ndarray, change_amount: bool = True, new_list_for_spatial_partition: bool = False) -> None:
        self._particles = value
        self._on_particles_change(change_amount, new_list_for_spatial_partition)
        
    def _on_particles_change(self, change_amount: bool = False, new_list_for_spatial_partition: bool = False) -> None:
        if change_amount:
            self.n_particles = self._particles.shape[0]
            
        self.populate_spatial_partition(create_new_list=new_list_for_spatial_partition)
        self.cache_densities(create_new_array=change_amount)

#################################################################################################
# Methods
#################################################################################################

    def get_partition_index_from_pos(self, point: np.ndarray) -> int:
        """ Get the index in the spatial partition, of the particle. """
        cell_x = int(point[0] // self.radius)
        cell_y = int(point[1] // self.radius)
        return cell_y * self.grid_width + cell_x

    def populate_spatial_partition(self, create_new_list: bool = False) -> None:
        """ Put the particles indices in their correct spatial partition. """
        if create_new_list:
            self.spatial_partition_list = [list() for _ in range(self.total_cells)]

        for particle_index, particle in enumerate(self._particles):
            cell_index = self.get_partition_index_from_pos(particle)
            self.spatial_partition_list[cell_index].append(particle_index)

    def create_neighboring_cells_array(self):
        """ Make array where each row are the indices of the cells neighboring the cell of that index. """
        self.neighboring_cells = np.full((self.total_cells, self.N_NEIGHBORS), -1, dtype=np.int32)
        for i in range(self.total_cells):
            # neighbor_cell_indices = self.neighbor_offsets + i
            neighbor_cell_indices = [
                i + off
                for off in self.neighbor_offsets
                if 0 <= (i + off) < self.total_cells
            ]
            n_neighbors = len(neighbor_cell_indices)
            self.neighboring_cells[i, 0:n_neighbors] = neighbor_cell_indices

    def get_neighboring_particle_indices(self, point: np.ndarray) -> np.ndarray:
        """ Returns the indices of the particles in the neighboring partitions. """
        partition_index = self.get_partition_index_from_pos(point)
        neighbor_cell_indices = self.neighboring_cells[partition_index]
        list_of_lists_of_particle_indices = [self.spatial_partition_list[i] for i in neighbor_cell_indices]
        # np.concatenate apparently doesn't work with empty lists
        non_empty_lists = [lst for lst in list_of_lists_of_particle_indices if lst]
        if not non_empty_lists:
            return np.array([], dtype=np.intp)
        # Could at some point return a floats instead of ints, so might have to .astype(np.int32)
        return np.concatenate(non_empty_lists, axis=0)

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


    @staticmethod
    def particle_grid(amount: int) -> np.ndarray:
        """This seems quite shit. """
        side_length = int(np.ceil(np.sqrt(amount)))
        spacing = DensityFluidSim.DEFAULT_RADIUS * 0.5
        points = []
        for i in range(amount):
            x = (i % side_length) * spacing + spacing
            y = (i // side_length) * spacing + spacing
            points.append([x, y])
        return np.array(points, dtype=np.float32)

#########################################################################################################################
    # Depreciated functions
#########################################################################################################################

    def density_contribution_from_cell(self, point: np.ndarray, cell_index: int) -> float:
        points_in_cell = self.spatial_partition_list[cell_index]
        diffs = points_in_cell - point
        dists = np.linalg.norm(diffs, axis=1)
        linear = np.maximum(0.0, self.radius - dists)
        influences = np.power(linear, 2)
        return float((influences.sum() * self.mass * self.inverse_volume))

    def _density_at_point_loop(self, point: np.ndarray) -> float:
        total_density = 0.0
        for other_point in self._particles:
            distance = float(np.linalg.norm(point - other_point))
            influence = square_kernel(distance, self._radius, self.inverse_volume)
            total_density += influence * self.mass
        return total_density

    def _density_at_point_vector(self, point: np.ndarray) -> float:
        diffs = self._particles - point
        dists = np.linalg.norm(diffs, axis=1)
        linear = np.maximum(0.0, self.radius - dists)
        influences = np.power(linear, 2)
        return float(influences.sum() * self.mass * self.inverse_volume)

    def cell_coords_from_index(self, index: int) -> np.ndarray:
        cell_y = index // self.grid_width
        cell_x = index % self.grid_width
        return np.array([cell_x, cell_y], dtype=np.int32)
        
    def neighboring_cell_indices(self, point: np.ndarray) -> list[int]:
        current_index = self.get_partition_index_from_pos(point)
        neighbor_cell_indices = [
            current_index + off
            for off in self.neighbor_offsets
            if 0 <= (current_index + off) < self.total_cells
        ]
        return neighbor_cell_indices

    def generate_random_particles_stack(self, num_points:int) -> np.ndarray:
        """ I mean read the method name. """
        max_width = self.sim_width
        max_height = self.sim_height
        x_coords = np.random.uniform(low=0, high=max_width, size=num_points)
        y_coords = np.random.uniform(low=0, high=max_height, size=num_points)
        return np.stack((x_coords, y_coords), axis=1)

"""
  
    def draw_density_image(self) -> None:
        print("I am in draw density image")
        density_image = normalize_array(self.DFS.density_image)
        # this looks fucking cool
        color_func_vectorized = np.vectorize(colormap_RWB, signature='(f4)->(3u1)')
        print("Until here")
        try:
            rgb_array = color_func_vectorized(density_image)
        except Exception as e:
            print(e)
        print(rgb_array)
        image_surface = pygame.surfarray.make_surface(rgb_array)
        self.screen.blit(image_surface, (0, 0))

"""

def main():
    particle_grid = DensityFluidSim.particle_grid(10)
    DFS = DensityFluidSim(particle_grid)
    random_particles = DFS.generate_random_particles(1000)
    DFS.set_particles(random_particles)


if __name__ == "__main__":
    main()

