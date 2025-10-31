
import numpy as np


class SpatialPartitionList:
    N_NEIGHBORS = 9
    
    sim_width: int
    sim_height: int
    total_cells: int
    grid_width: int
    grid_height: int
    
    partition_list: list
    neighboring_cells: np.ndarray
    
#########################################################################################################################
# Special Methods
#########################################################################################################################
    
    def __init__(self,
        cell_size: int,
        sim_width: int,
        sim_height: int,
        ):
        
        self.sim_width = sim_width
        self.sim_height = sim_height
        
        self.cell_size = cell_size
        
        self.create_new_partition_list()
        
    def __getitem__(self, key: int):
        """ Allows instance[index] or instance[slice] to access elements of self.main_list. """
        return self.partition_list[key]
    
    def __iter__(self):
        """Returns an iterator over the internal list."""
        return iter(self.partition_list)
    
#########################################################################################################################
# Properties
#########################################################################################################################
    
    @property
    def cell_size(self) -> float:
        return self._cell_size
    
    @cell_size.setter
    def cell_size(self, value: float) -> None: 
        self._cell_size = value
        self._calc_vals_dependent_on_cell_size()

    def _calc_vals_dependent_on_cell_size(self) -> None:
        # Grid dimensions
        self.grid_width = int(self.sim_width // self.cell_size)
        self.grid_height = int(self.sim_height // self.cell_size)
        self.total_cells = self.grid_width * self.grid_height
        # 
        self.neighbor_offsets = np.array([
            -self.grid_width -1, -self.grid_width, -self.grid_width +1,
            -1,                 0,              1,
            self.grid_width -1,  self.grid_width, self.grid_width +1
        ])
        self.create_neighboring_cells_array()
        print(self.neighboring_cells)

#########################################################################################################################
# Instance Methods
#########################################################################################################################
    
    def get_partition_index_from_pos(self, point: np.ndarray) -> int:
        """ Get the index in the spatial partition, of the particle. """
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        return cell_y * self.grid_width + cell_x

    def create_new_partition_list(self):
        """ Read function name. """
        self.partition_list = [list() for _ in range(self.total_cells)]

    def populate_spatial_partition(self, particles: np.ndarray, create_new_list: bool = False) -> None:
        """ Put the particles indices in their correct spatial partition. """
        if create_new_list:
            self.create_new_partition_list()

        for particle_index, particle in enumerate(particles):
            cell_index = self.get_partition_index_from_pos(particle)
            self.partition_list[cell_index].append(particle_index)

    def create_neighboring_cells_array(self):
        """ Create array where each row are the indices of the cells neighboring the cell of that index """
        self.neighboring_cells = np.full((self.total_cells, self.N_NEIGHBORS), -1, dtype=np.int32)
        for i in range(self.total_cells):
            # neighbor_cell_indices = self.neighbor_offsets + i
            current_col = i % self.grid_width
            last_col = self.grid_width - 1
            
            if current_col == 0:
                offsets = self.neighbor_offsets[[1, 2, 4, 5, 7, 8]]
            elif current_col == last_col:
                offsets = self.neighbor_offsets[[0, 1, 3, 4, 6, 7]]
            else:
                offsets = self.neighbor_offsets     
            
            neighbor_cell_indices = [
                i + offset
                for offset in offsets
                if 0 <= (i + offset) < self.total_cells
            ]
            n_neighbors = len(neighbor_cell_indices)
            self.neighboring_cells[i, 0:n_neighbors] = neighbor_cell_indices

    def get_neighboring_particle_indices(self, point: np.ndarray) -> np.ndarray:
        """ Returns the indices of the particles in the neighboring partitions. """
        partition_index = self.get_partition_index_from_pos(point)
        neighbor_cell_indices = self.neighboring_cells[partition_index]
        list_of_lists_of_particle_indices = [self.partition_list[i] for i in neighbor_cell_indices]
        # np.concatenate apparently doesn't work with empty lists
        non_empty_lists = [lst for lst in list_of_lists_of_particle_indices if lst]
        if not non_empty_lists:
            return np.array([], dtype=np.intp)
        # Could at some point return a floats instead of ints, so might have to .astype(np.int32)
        return np.concatenate(non_empty_lists, axis=0)


