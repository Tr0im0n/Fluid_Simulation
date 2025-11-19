
import os
import sys
import numpy as np
import pygame

from src.logic import flat_calcer
from src.logic.particle_system import ParticleSystem
from src.logic.spatial_partition_list import SpatialPartitionList

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_relative = os.path.join(current_dir, '..', '..')
project_root = os.path.normpath(project_root_relative)
sys.path.append(project_root)

from src.logic.density import DensityFluidSim
from src.logic.looping_circle import LoopingCircle
from src.utils.colors import BLACK, colormap, colormap_RWB, DARKGRAY, colormap_array_BWR, colormap_array_BWR_optimized, colormap_array_mpl
from src.utils.arrays import normalize_array

print("Finished imports")

class Game:

    def __init__(self, 
            width: int = 800, 
            height: int = 600, 
            fps: int = 60):

        self.width = width
        self.height = height
        self.fps = fps

        pygame.init()
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Basic Pygame Loop")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.font = pygame.font.SysFont("arial", 24)

        self.h = 61.2
        self.rho_0 = 100

        self.ps = ParticleSystem.from_random_particles(1000, 2, width, height, 42)
        # self.spl = SpatialPartitionList.with_particles(self.ps.positions, h, width, height)
        self.spl = SpatialPartitionList(self.h, width, height)



    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def update(self, dt: float):
        print("Entered update")
        self.spl.populate(self.ps.positions)
        a_pressure = flat_calcer.calc_things(self.ps.positions, self.h, self.spl, self.rho_0)
        print("update 2")
        pressure_multi = 1
        self.ps.velocities -= pressure_multi * a_pressure * dt
        self.ps.positions += self.ps.velocities * dt
        flat_calcer.apply_boundary_collision(self.ps.positions, self.ps.velocities, self.width, self.height)
        
        
    def draw_particles(self):
        for point in self.ps.positions:
            pygame.draw.circle(self.screen, (255, 255, 255), (int(point[0]), int(point[1])), 2)
            
    # def draw_particles_color_density(self):
    #     densities = normalize_array(self.DFS.densities_of_particles)
    #     colors = [colormap(density, (0, 0, 255), (255, 255, 255), (255, 0, 0)) for density in densities]
    #     for point, color in zip(self.DFS.particles, colors):
    #         pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 3)

    def draw_fps(self):
        current_fps = self.clock.get_fps()
        fps_text = f"FPS: {current_fps:.2f}"
        fps_font = self.font.render(fps_text, True, (200, 200, 200))
        self.screen.blit(fps_font, (10, 10))
        
    def draw_runtime(self):
        runtime = pygame.time.get_ticks()/1000
        time_text = f"Runtime: {runtime:.2f} s"
        time_font = self.font.render(time_text, True, (200, 200, 200))
        self.screen.blit(time_font, (10, 40))

    def draw(self):
        self.screen.fill((0, 0, 0))
        # self.draw_particles_color_density()
        self.draw_particles()
        self.draw_fps()
        self.draw_runtime()

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(self.fps) / 1000.0  # seconds since last frame
                self.handle_events()
                self.update(dt)
                self.draw()
                pygame.display.flip()
        finally:
            pygame.quit()
            sys.exit()


def main():
    game1 = Game()
    game1.run()


if __name__ == "__main__":
    print("Start game loop")
    main()
    
    
    
    