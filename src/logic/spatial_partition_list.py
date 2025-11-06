
import math
import numpy as np
from numpy.typing import NDArray

list_list_int = list[list[int]] # not used yet

class SpatialPartitionList:
    """ Can't be changed after instantiation. """
    N_NEIGHBORS = 9
    
    CELL_TO_CELL_NEIGHBORS: np.ndarray
    cell_to_particle_neighbors: list[np.ndarray]
    
#########################################################################################################################
# Special Methods
#########################################################################################################################
    
    def __init__(self,
            particles: np.ndarray | None,
            cell_size: float,
            sim_width: int,
            sim_height: int,
        ):
        
        self.cell_size = cell_size
        self.sim_width = sim_width
        self.sim_height = sim_height
        
        grid_vals = self.calculate_grid_vals(sim_width, sim_height, cell_size)
        self.grid_width, self.grid_height, self.total_cells, self.neighbor_offsets = grid_vals
        self.create_neighboring_cells_array()
        
        self._partition_list = [list() for _ in range(self.total_cells)]
        if particles is not None:
            self.populate(particles)
        self.create_neighboring_particles_array()
        
    def __getitem__(self, key: int):
        """ Allows instance[index] or instance[slice] to access elements of self.main_list. """
        return self._partition_list[key]
    
    def __iter__(self):
        """Returns an iterator over the internal list."""
        return iter(self._partition_list)

#################################################################################################
# Instance Methods
#################################################################################################

    def create_neighboring_cells_array(self) -> None:
        """ Create array where each row are the indices of the cells neighboring the cell of that index. 
            Only used once in the init. """
        self.CELL_TO_CELL_NEIGHBORS = self.calc_cell_to_cell_neighbors(
            self.neighbor_offsets, self.grid_width, self.total_cells)
        
    def get_partition_index_from_pos(self, point: np.ndarray) -> int:
        """ Get the index in the spatial partition, of the particle. """
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        return cell_y * self.grid_width + cell_x

    def populate(self, particles: np.ndarray) -> None:
        """ Used in the init. 
            And can by used outside! """
        for particle_index, particle in enumerate(particles):
            cell_index = self.get_partition_index_from_pos(particle)
            self._partition_list[cell_index].append(particle_index)
        self.create_neighboring_particles_array()
    
    def create_neighboring_particles_array(self) -> None:
        """set self.CELL_TO_PARTICLE_NEIGHBORS"""
        self.cell_to_particle_neighbors = self.calc_cell_to_particle_neighbors(
            self._partition_list, self.CELL_TO_CELL_NEIGHBORS)

#################################################################################################
# Static Methods
#################################################################################################

    @staticmethod
    def calculate_grid_vals(sim_width: int, sim_height: int, cell_size: float) -> tuple:
        """ 
        Calculates all grid-related constants necessary for spatial partitioning.
        
        Returns: grid_width, grid_height, total_cells, neighbor_offsets
        """
        # Grid dimensions
        grid_width = math.ceil(sim_width / cell_size)
        grid_height = math.ceil(sim_height / cell_size)
        total_cells = grid_width * grid_height
        # Neighbor info
        neighbor_offsets = np.array([
            -grid_width - 1, -grid_width, -grid_width + 1,
            -1,                0,             1,
            grid_width - 1,  grid_width,  grid_width + 1
        ], dtype=np.int32)
        # return tuple
        return grid_width, grid_height, total_cells, neighbor_offsets
    
    @staticmethod
    def calc_neighbor_indices_of_cell(cell_index: int, neighbor_offsets: np.ndarray, grid_width: int, total_cells) -> np.ndarray:
        """ Only used in next method. """
        current_col = cell_index % grid_width
        last_col = grid_width - 1
        
        if current_col == 0:
            offsets = neighbor_offsets[[1, 2, 4, 5, 7, 8]]
        elif current_col == last_col:
            offsets = neighbor_offsets[[0, 1, 3, 4, 6, 7]]
        else:
            offsets = neighbor_offsets     
        
        potential_indices = cell_index + offsets
        mask = (potential_indices >= 0) & (potential_indices < total_cells)
        return potential_indices[mask]

    @classmethod
    def calc_cell_to_cell_neighbors(cls, neighbor_offsets: np.ndarray, grid_width: int, total_cells) -> np.ndarray:
        """ Create array where each row are the indices of the cells neighboring the cell of that index. """
        n_neighbors = cls.N_NEIGHBORS
        answer = np.full((total_cells, n_neighbors), -1, dtype=np.int32)
        for cell_index in range(total_cells):
            neighbor_cell_indices = cls.calc_neighbor_indices_of_cell(cell_index, neighbor_offsets, grid_width, total_cells)
            n_neighbors = len(neighbor_cell_indices)
            answer[cell_index, 0:n_neighbors] = neighbor_cell_indices
        return answer
    
    @staticmethod
    def calc_cell_to_particle_neighbors(spl: list[list[int]], 
                                       cell_to_cell_neighbors: NDArray[np.int32]) -> list[NDArray[np.int32]]:
        """ Returns the indices of the particles in the neighboring partitions. """
        result = [
            np.array([
                particle_index
                for cell_index in neighbor_group
                for particle_index in spl[cell_index]
            ], dtype=np.int32)
            for neighbor_group in cell_to_cell_neighbors
        ]
        return result
    
    
""" 

    def _create_neighbor_indices_of_cell(self, cell_index: int) -> np.ndarray:
        # Only used in next method.
        current_col = cell_index % self.GRID_WIDTH
        last_col = self.GRID_WIDTH - 1
        
        if current_col == 0:
            offsets = self.NEIGHBOR_OFFSETS[[1, 2, 4, 5, 7, 8]]
        elif current_col == last_col:
            offsets = self.NEIGHBOR_OFFSETS[[0, 1, 3, 4, 6, 7]]
        else:
            offsets = self.NEIGHBOR_OFFSETS     
        
        potential_indices = cell_index + offsets
        mask = (potential_indices >= 0) & (potential_indices < self.TOTAL_CELLS)
        return potential_indices[mask]

    def get_neighboring_particle_indices(self, point: np.ndarray) -> np.ndarray:
        # Returns the indices of the particles in the neighboring partitions. 
        partition_index = self.get_partition_index_from_pos(point)
        neighbor_cell_indices = self.NEIGHBORING_CELLS_ARRAY[partition_index]
        list_of_lists_of_particle_indices = [self._partition_list[i] for i in neighbor_cell_indices]
        # np.concatenate apparently doesn't work with empty lists
        non_empty_lists = [lst for lst in list_of_lists_of_particle_indices if lst]
        if not non_empty_lists:
            return np.array([], dtype=np.intp)
        # Could at some point return a floats instead of ints, so might have to .astype(np.int32)
        return np.concatenate(non_empty_lists, axis=0)
    
    def _calc_vals(self) -> None:
        # Only used in the init. 
        # Grid dimensions
        self.GRID_WIDTH = math.ceil(self.SIM_WIDTH / self.CELL_SIZE)
        self.GRID_HEIGHT = math.ceil(self.SIM_HEIGHT / self.CELL_SIZE)
        self.TOTAL_CELLS = self.GRID_WIDTH * self.GRID_HEIGHT
        # neighbor info
        self.NEIGHBOR_OFFSETS = np.array([
            -self.GRID_WIDTH -1, -self.GRID_WIDTH, -self.GRID_WIDTH +1,
            -1,                 0,              1,
            self.GRID_WIDTH -1,  self.GRID_WIDTH, self.GRID_WIDTH +1
        ])

"""    
    