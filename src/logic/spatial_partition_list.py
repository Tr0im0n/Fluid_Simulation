
import math
import numpy as np
from numpy.typing import NDArray

"""
Hilarious
This class has: 
1 init
1 alternate constructor
1 actual method
5 helper functions
"""

class SpatialPartitionList:
    """ Can't be changed after instantiation.
        If you need a new radius, make a new instance. """
    N_NEIGHBORS = 9
    N_PER_CELL = 8.0
    
    cell_to_particle_neighbor_indices: list[NDArray[np.int32]]
    amounts_of_interactions: NDArray[np.int32]
    center_indices_flat: NDArray[np.int32]
    neighbor_indices_flat: NDArray[np.int32]
    start_indices: NDArray[np.int32]
    
#########################################################################################################################
# Special Methods
#########################################################################################################################
    
    def __init__(self,
            cell_size: float,
            sim_width: int,
            sim_height: int):
        """ Init just the size, give particles later. """
        # self.particles = None
        self.cell_size = cell_size
        self.sim_width = sim_width
        self.sim_height = sim_height
        
        self.grid_columns = math.ceil(sim_width / cell_size)
        self.grid_rows = math.ceil(sim_height / cell_size)
        self.total_cells = self.grid_columns * self.grid_rows
        print(f"Total cells : {self.total_cells}")
        
        self.partition_list = []
        self.neighbor_cell_offsets = self.calc_neighbor_cell_offsets(self.grid_columns)
        self.cell_to_cell_neighbors = self.calc_cell_to_cell_neighbors(
                self.neighbor_cell_offsets, self.grid_columns, self.total_cells)

    @classmethod
    def with_particles(cls,
            particles: NDArray[np.float32],
            cell_size: float,
            sim_width: int,
            sim_height: int):
        """ Alternate constructor """
        instance = cls(cell_size, sim_width, sim_height)
        instance.populate(particles)
        return instance

#################################################################################################
# Instance Methods
#################################################################################################

# TODO make an in place method for this

    def populate(self, particles: np.ndarray) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
        """ Main method 
            returns: self.center_indices_flat, self.neighbor_indices_flat, self.start_indices """
        # self.particles = particles
        # Makes an array where each entry is the partition cell for that particle
        self.partition_indices = self.calc_partition_indices(
                particles, self.cell_size, self.grid_columns)
        # need to empty the list first here
        self.partition_list = [list() for _ in range(self.total_cells)]
        # Actually populating the partition list
        for particle_index, cell_index in enumerate(self.partition_indices):
            if cell_index >= self.total_cells:
                print(cell_index)
            self.partition_list[cell_index].append(particle_index)
        # Making big data structures for fast computation
        self.cell_to_particle_neighbor_indices = self.calc_cell_to_particle_neighbor_indices(
                self.partition_list, self.cell_to_cell_neighbors)
        
        cell_to_particle_neighbor_amounts = np.array([len(i) for i in self.cell_to_particle_neighbor_indices], dtype=np.int32)
        self.amounts_of_interactions = np.take(cell_to_particle_neighbor_amounts, self.partition_indices)
        self.center_indices_flat = np.repeat(np.arange(len(particles)), self.amounts_of_interactions)
        
        neighbor_indices_jagged = [self.cell_to_particle_neighbor_indices[i] for i in self.partition_indices]
        self.neighbor_indices_flat = np.concatenate(neighbor_indices_jagged)
        
        end_indices = np.cumsum(self.amounts_of_interactions)
        self.start_indices = np.concatenate(([0], end_indices[:-1]))
        
        return self.center_indices_flat, self.neighbor_indices_flat, self.start_indices

    def get_flat_vals(self):
        """ returns: self.center_indices_flat, self.neighbor_indices_flat, self.start_indices \n
            center_indices_flat, neighbor_indices_flat, start_indices """
        return self.center_indices_flat, self.neighbor_indices_flat, self.start_indices


#################################################################################################
# Static Methods
#################################################################################################

    @staticmethod
    def calc_neighbor_cell_offsets(grid_columns: int):
        neighbor_offsets = np.array([
            -grid_columns - 1, -grid_columns, -grid_columns + 1,
            -1,                0,             1,
            grid_columns - 1,  grid_columns,  grid_columns + 1
        ], dtype=np.int32)
        return neighbor_offsets
    
    @staticmethod
    def calc_partition_indices(points: np.ndarray, cell_size: float, grid_columns: int) -> NDArray[np.int32]:
        """ Get the index in the spatial partition, of the particles. """
        cell_x = np.floor_divide(points[:, 0], cell_size).astype(np.int32)
        cell_y = np.floor_divide(points[:, 1], cell_size).astype(np.int32)
        return cell_y * grid_columns + cell_x
    
    @staticmethod
    def calc_neighbor_indices_of_cell(cell_index: int, neighbor_offsets: NDArray[np.int32], grid_width: int, total_cells: int) -> NDArray[np.int32]:
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
    def calc_cell_to_cell_neighbors(cls, neighbor_offsets: NDArray[np.int32], grid_width: int, total_cells: int) -> NDArray[np.int32]:
        """ Create array where each row are the indices of the cells neighboring the cell of that index. """
        n_neighbors = cls.N_NEIGHBORS
        answer = np.full((total_cells, n_neighbors), -1, dtype=np.int32)
        for cell_index in range(total_cells):
            neighbor_cell_indices = cls.calc_neighbor_indices_of_cell(cell_index, neighbor_offsets, grid_width, total_cells)
            n_neighbors = len(neighbor_cell_indices)
            answer[cell_index, 0:n_neighbors] = neighbor_cell_indices
        return answer
    
    @staticmethod
    def calc_cell_to_particle_neighbor_indices(spl: list[list[int]], 
                                       cell_to_cell_neighbors: NDArray[np.int32]) -> list[NDArray[np.int32]]:
        """ Returns the indices of the particles in the neighboring partitions. """
        result = [
            np.concatenate(
                [np.asarray(spl[cell_index], dtype=np.int32)
                    for cell_index in neighbor_group]
            )
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
        
        
        
    def get_partition_index_from_pos(self, point: np.ndarray, ) -> int:
        # Get the index in the spatial partition, of the particle. 
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        return cell_y * self.grid_columns + cell_x
        

    @staticmethod
    def calculate_grid_vals(sim_width: int, sim_height: int, cell_size: float) -> tuple:
        
        # Calculates all grid-related constants necessary for spatial partitioning.
        
        # Returns: grid_columns, grid_rows, total_cells, neighbor_offsets
        
        # Grid dimensions
        grid_columns = math.ceil(sim_width / cell_size)
        grid_rows = math.ceil(sim_height / cell_size)
        total_cells = grid_columns * grid_rows
        # Neighbor info
        neighbor_offsets = np.array([
            -grid_columns - 1, -grid_columns, -grid_columns + 1,
            -1,                0,             1,
            grid_columns - 1,  grid_columns,  grid_columns + 1
        ], dtype=np.int32)
        # return tuple
        return grid_columns, grid_rows, total_cells, neighbor_offsets


    def create_cell_to_cell_neighbors(self) -> None:
        # Create array where each row are the indices of the cells neighboring the cell of that index. 
        #    Only used once in the init. 
        self.cell_to_cell_neighbors = self.calc_cell_to_cell_neighbors(
            self.neighbor_cell_offsets, self.grid_columns, self.total_cells)
            
    def create_cell_to_particle_neighbor_indices(self) -> None:
        # set self.CELL_TO_PARTICLE_NEIGHBORS
        self.cell_to_particle_neighbor_indices = self.calc_cell_to_particle_neighbor_indices(
            self._partition_list, self.cell_to_cell_neighbors)

    def create_particle_to_particle_neighbor_indices(self) -> None:
        # I mean just read the function name. 
        self.particle_to_particle_neighbor_indices = self.calc_particle_to_particle_neighbor_indices(
            self.cell_to_particle_neighbor_indices, self.partition_indices)
            
    @staticmethod
    def calc_particle_to_particle_neighbor_indices(cell_to_particle_neighbor_indices: list[NDArray[np.int32]], 
                                                   partition_indices: NDArray[np.int32]) -> NDArray[np.int32]:
        # return np.take(cell_to_particle_neighbor_indices, partition_indices) # Jagged array error
        temp_list = [cell_to_particle_neighbor_indices[i] for i in partition_indices]
        len_array = np.array([len(i) for i in temp_list], dtype=np.int32)
        return np.concatenate(temp_list)  
            
    
        
    def __getitem__(self, key: int):
        # Allows instance[index] or instance[slice] to access elements of self.main_list. 
        return self._partition_list[key]
    
    def __iter__(self):
        # Returns an iterator over the internal list.
        return iter(self._partition_list)       
            

"""    
    