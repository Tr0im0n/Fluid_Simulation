
import numpy as np


class SpatialPartitionList:
    """ Can't be changed after instantiation. """
    N_NEIGHBORS = 9
    
    grid_width: int
    grid_height: int
    total_cells: int
    
    neighboring_cells: np.ndarray
    
#########################################################################################################################
# Special Methods
#########################################################################################################################
    
    def __init__(self,
            particles: np.ndarray,
            cell_size: int,
            sim_width: int,
            sim_height: int,
        ):
        
        self._partition_list = []
        
        self.cell_size = cell_size
        self.sim_width = sim_width
        self.sim_height = sim_height
        
        self.calc_vals()
        self.create_neighboring_cells_array()
        self.populate_spatial_partition(particles)
        
        
    def __getitem__(self, key: int):
        """ Allows instance[index] or instance[slice] to access elements of self.main_list. """
        return self._partition_list[key]
    
    def __iter__(self):
        """Returns an iterator over the internal list."""
        return iter(self._partition_list)

#########################################################################################################################
# Instance Methods
#########################################################################################################################
    
    def calc_vals(self) -> None:
        """ Only used once in the init. """
        # Grid dimensions
        self.grid_width = int(self.sim_width // self.cell_size)
        self.grid_height = int(self.sim_height // self.cell_size)
        self.total_cells = self.grid_width * self.grid_height
        # neighbor info
        self.neighbor_offsets = np.array([
            -self.grid_width -1, -self.grid_width, -self.grid_width +1,
            -1,                 0,              1,
            self.grid_width -1,  self.grid_width, self.grid_width +1
        ])
        
    def get_partition_index_from_pos(self, point: np.ndarray) -> int:
        """ Get the index in the spatial partition, of the particle. """
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        return cell_y * self.grid_width + cell_x

    def populate_spatial_partition(self, particles: np.ndarray) -> None:
        """ Only used once in the init. """
        for particle_index, particle in enumerate(particles):
            cell_index = self.get_partition_index_from_pos(particle)
            self._partition_list[cell_index].append(particle_index)

    def calc_neighbor_indices_of_cell(self, cell_index: int):
        """ Only used in next method. """
        current_col = cell_index % self.grid_width
        last_col = self.grid_width - 1
        
        if current_col == 0:
            offsets = self.neighbor_offsets[[1, 2, 4, 5, 7, 8]]
        elif current_col == last_col:
            offsets = self.neighbor_offsets[[0, 1, 3, 4, 6, 7]]
        else:
            offsets = self.neighbor_offsets     
        
        potential_indices = cell_index + offsets
        mask = (potential_indices >= 0) & (potential_indices < self.total_cells)
        return potential_indices[mask]

    def create_neighboring_cells_array(self):
        """ Create array where each row are the indices of the cells neighboring the cell of that index. 
            Only used once in the init. """
        self.neighboring_cells = np.full((self.total_cells, self.N_NEIGHBORS), -1, dtype=np.int32)
        for cell_index in range(self.total_cells):
            neighbor_cell_indices = self.calc_neighbor_indices_of_cell(cell_index)
            n_neighbors = len(neighbor_cell_indices)
            self.neighboring_cells[cell_index, 0:n_neighbors] = neighbor_cell_indices

    def get_neighboring_particle_indices(self, point: np.ndarray) -> np.ndarray:
        """ Returns the indices of the particles in the neighboring partitions. """
        partition_index = self.get_partition_index_from_pos(point)
        neighbor_cell_indices = self.neighboring_cells[partition_index]
        list_of_lists_of_particle_indices = [self._partition_list[i] for i in neighbor_cell_indices]
        # np.concatenate apparently doesn't work with empty lists
        non_empty_lists = [lst for lst in list_of_lists_of_particle_indices if lst]
        if not non_empty_lists:
            return np.array([], dtype=np.intp)
        # Could at some point return a floats instead of ints, so might have to .astype(np.int32)
        return np.concatenate(non_empty_lists, axis=0)


