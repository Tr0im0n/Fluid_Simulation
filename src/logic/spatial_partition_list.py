
import numpy as np


class SpatialPartitionList:
    """ Can't be changed after instantiation. """
    N_NEIGHBORS = 9
    
    GRID_WIDTH: int
    GRID_HEIGHT: int
    TOTAL_CELLS: int
    
    NEIGHBORING_CELLS_ARRAY: np.ndarray
    
#########################################################################################################################
# Special Methods
#########################################################################################################################
    
    def __init__(self,
            particles: np.ndarray,
            cell_size: float,
            sim_width: int,
            sim_height: int,
        ):
        
        self._partition_list = []
        
        self.CELL_SIZE = cell_size
        self.SIM_WIDTH = sim_width
        self.SIM_HEIGHT = sim_height
        
        self._calc_vals()
        self._create_neighboring_cells_array()
        self.populate_spatial_partition(particles)
        
        
    def __getitem__(self, key: int):
        """ Allows instance[index] or instance[slice] to access elements of self.main_list. """
        return self._partition_list[key]
    
    def __iter__(self):
        """Returns an iterator over the internal list."""
        return iter(self._partition_list)

#################################################################################################
# Instance Methods
#################################################################################################
    
    def _calc_vals(self) -> None:
        """ Only used in the init. """
        # Grid dimensions
        self.GRID_WIDTH = int(self.SIM_WIDTH // self.CELL_SIZE)
        self.GRID_HEIGHT = int(self.SIM_HEIGHT // self.CELL_SIZE)
        self.TOTAL_CELLS = self.GRID_WIDTH * self.GRID_HEIGHT
        # neighbor info
        self.NEIGHBOR_OFFSETS = np.array([
            -self.GRID_WIDTH -1, -self.GRID_WIDTH, -self.GRID_WIDTH +1,
            -1,                 0,              1,
            self.GRID_WIDTH -1,  self.GRID_WIDTH, self.GRID_WIDTH +1
        ])

    def _create_neighbor_indices_of_cell(self, cell_index: int):
        """ Only used in next method. """
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

    def _create_neighboring_cells_array(self):
        """ Create array where each row are the indices of the cells neighboring the cell of that index. 
            Only used once in the init. """
        self.NEIGHBORING_CELLS_ARRAY = np.full((self.TOTAL_CELLS, self.N_NEIGHBORS), -1, dtype=np.int32)
        for cell_index in range(self.TOTAL_CELLS):
            neighbor_cell_indices = self._create_neighbor_indices_of_cell(cell_index)
            n_neighbors = len(neighbor_cell_indices)
            self.NEIGHBORING_CELLS_ARRAY[cell_index, 0:n_neighbors] = neighbor_cell_indices
        
    def get_partition_index_from_pos(self, point: np.ndarray) -> int:
        """ Get the index in the spatial partition, of the particle. """
        cell_x = int(point[0] // self.CELL_SIZE)
        cell_y = int(point[1] // self.CELL_SIZE)
        return cell_y * self.GRID_WIDTH + cell_x

    def populate_spatial_partition(self, particles: np.ndarray) -> None:
        """ Used in the init. 
            And can by used outside! """
        self._partition_list = []
        for particle_index, particle in enumerate(particles):
            cell_index = self.get_partition_index_from_pos(particle)
            self._partition_list[cell_index].append(particle_index)

    def get_neighboring_particle_indices(self, point: np.ndarray) -> np.ndarray:
        """ Returns the indices of the particles in the neighboring partitions. """
        partition_index = self.get_partition_index_from_pos(point)
        neighbor_cell_indices = self.NEIGHBORING_CELLS_ARRAY[partition_index]
        list_of_lists_of_particle_indices = [self._partition_list[i] for i in neighbor_cell_indices]
        # np.concatenate apparently doesn't work with empty lists
        non_empty_lists = [lst for lst in list_of_lists_of_particle_indices if lst]
        if not non_empty_lists:
            return np.array([], dtype=np.intp)
        # Could at some point return a floats instead of ints, so might have to .astype(np.int32)
        return np.concatenate(non_empty_lists, axis=0)


