
import numpy as np
from numpy.typing import NDArray

class ParticleSystem:
    """
    Manages all particle data using a Structure of Arrays
    """
    def __init__(self, 
                n_particles: int, 
                n_dimensions: int=2,
                *args,
                positions: None | NDArray[np.float32] = None,
                masses = 1,
                **kwargs):
        
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions

        self.positions = positions if positions is not None else np.ones((2, 2), dtype=np.float32)
        self.velocities = np.zeros((n_particles, n_dimensions), dtype=np.float32)
        self.masses = masses
        self.densities = np.empty(n_particles, dtype=np.float32)
        self.pressures_normalized = np.empty(n_particles, dtype=np.float32)

    @classmethod
    def from_random_particles(cls, 
                n_particles: int, n_dimensions: int=2,
                max_width: float = 100., max_height: float = 100., 
                seed: int = 42):
        """ I mean read the method name. """
        np.random.seed(seed)
        positions = np.random.uniform(
            low=[0, 0], 
            high=[max_width, max_height], 
            size=(n_particles, n_dimensions)
        ).astype(np.float32)
        return ParticleSystem(n_particles, n_dimensions, positions=positions)
        
        



