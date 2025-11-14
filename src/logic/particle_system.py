
import numpy as np

class ParticleSystem:
    """
    Manages all particle data using a Structure of Arrays
    """
    def __init__(self, 
                N_particles: int, 
                dimensions: int=2,
                *args,
                positions = None,
                masses = 1,
                **kwargs):
        
        self.N_particles = N_particles
        self.dimensions = dimensions

        self.positions = positions
        self.velocities = np.zeros((N_particles, dimensions), dtype=np.float32)
        self.masses = masses
        self.densities = np.empty(N_particles, dtype=np.float32)
    





