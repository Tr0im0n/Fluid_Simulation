
import numpy as np
from numpy.typing import NDArray

from src.logic.particle_system import ParticleSystem
from src.logic.spatial_partition_list import SpatialPartitionList


def calc_things(ps: ParticleSystem, spl: SpatialPartitionList, 
                h: float, rho_0: float, gamma: int = 5, c_0_2: float = 100):
    epsilon = 1e-9
    PI = np.pi
    h2 = h**2
    h5 = h**5
    h8 = h**8
    
    alpha_poly6 = 4.0 / (PI * h8)
    alpha_spiky = 10.0 / (PI * h5)
    alpha_viscosity = 40.0 / (PI * h5)
    
    center_indices_flat, neighbor_indices_flat, start_indices = spl.get_flat_vals()
    center_poss = np.take(ps.positions, center_indices_flat, axis=0)
    neighbor_poss = np.take(ps.positions, neighbor_indices_flat, axis=0)

    r_ij = center_poss - neighbor_poss
    r_sq = np.sum(np.square(r_ij), axis=1)
    r = np.sqrt(r_sq)
    
    w_poly6 = np.power(np.maximum(h2 - r_sq, 0), 3)
    w_viscosity = np.maximum(h - r, 0)
    
    unit_vectors = r_ij / (r[:, np.newaxis] + epsilon)
    w_spiky_magnitude = np.square(w_viscosity)
    w_spiky = w_spiky_magnitude[:, np.newaxis] * unit_vectors

    densities = np.add.reduceat(w_poly6, start_indices) * alpha_poly6
    
    # pressures_normalized = (densities - rho_0) / densities**2
    b = c_0_2 * rho_0 / gamma
    pressures_normalized = b / (densities**2) * (np.pow(densities / rho_0, gamma) - 1) 
    
    center_pressures = np.take(pressures_normalized, center_indices_flat)
    neighbor_pressures = np.take(pressures_normalized, neighbor_indices_flat)
    
    a_pressure_flat = (center_pressures + neighbor_pressures)[:, np.newaxis] * w_spiky
    a_pressure = np.add.reduceat(a_pressure_flat, start_indices) * alpha_spiky
    
    center_velocities = np.take(ps.velocities, center_indices_flat)
    neighbor_velocities = np.take(ps.velocities, neighbor_indices_flat)
    neighbor_densities = np.take(densities, neighbor_indices_flat)
    
    # a_viscosity_flat = (neighbor_velocities - center_velocities) / neighbor_densities[:, np.newaxis] * w_viscosity
    # a_viscosity = np.add.reduceat(a_viscosity_flat, start_indices) * alpha_viscosity / densities[:, np.newaxis]
    
    return a_pressure, a_viscosity, densities
    
    
def apply_boundary_collision(positions: NDArray[np.float32], velocities: NDArray[np.float32], 
                             max_width: float, max_height: float, 
                             bounce_factor: float = 0.5, epsilon: float = 1e-5) -> None:
    left_hit = positions[:, 0] < 0.0
    right_hit = positions[:, 0] > max_width
    top_hit = positions[:, 1] < 0.0
    bottom_hit = positions[:, 1] > max_height
    
    left_right_hit = left_hit | right_hit
    top_bottom_hit = top_hit | bottom_hit
    
    positions[left_hit, 0] = epsilon
    positions[right_hit, 0] = max_width - epsilon
    positions[top_hit, 1] = epsilon
    positions[bottom_hit, 1] = max_height - epsilon
    
    velocities[left_right_hit, 0] *= -bounce_factor
    velocities[top_bottom_hit, 1] *= -bounce_factor
    
    
    
    
    
    