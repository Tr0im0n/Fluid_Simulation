
import os
import sys
import numpy as np
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_relative = os.path.join(current_dir, '..', '..')
project_root = os.path.normpath(project_root_relative)
sys.path.append(project_root)

from src.logic.density import DensityFluidSim
from src.logic.looping_circle import LoopingCircle
from src.utils.colors import BLACK, colormap, colormap_RWB, DARKGRAY, colormap_array_BWR
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
        
        self.looping_circle = LoopingCircle(width, height)
        self.font = pygame.font.SysFont("arial", 24)

        self.DFS = DensityFluidSim(None, 790, 590, 45.2)
        # self.DFS.set_particles(self.DFS.generate_random_particles(1000, seed=7))


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def update(self, dt: float):
        self.looping_circle.update(dt)

    def draw_particles(self):
        densities = normalize_array(self.DFS.cached_densities)
        for point in self.DFS.particles:
            pygame.draw.circle(self.screen, (0, 0, 0), (int(point[0]), int(point[1])), 1)
            
    def draw_particles_color_density(self):
        densities = normalize_array(self.DFS.cached_densities)
        colors = [colormap(density, (0, 0, 255), (255, 255, 255), (255, 0, 0)) for density in densities]
        for point, color in zip(self.DFS.particles, colors):
            pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 3)
            
    def draw_density_image(self) -> None:
        density_image = normalize_array(self.DFS.density_image)
        # this looks fucking cool
        # color_func_vectorized = np.vectorize(colormap_RWB, signature='(f4)->(3u1)')
        rgb_array = colormap_array_BWR(density_image)
        transposed_rgb_array = rgb_array.transpose(1, 0, 2)
        image_surface = pygame.surfarray.make_surface(transposed_rgb_array)
        self.screen.blit(image_surface, (0, 0))

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
        # self.looping_circle.draw(self.screen)
        # self.draw_particles_color_density()
        self.draw_density_image()
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
