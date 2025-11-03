
import os
import sys
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_relative = os.path.join(current_dir, '..', '..')
project_root = os.path.normpath(project_root_relative)
sys.path.append(project_root)

from src.logic.density import DensityFluidSim
from src.logic.looping_circle import LoopingCircle
from src.utils.colors import colormap

print("Finished imports")

class Game:
    fps: int

    DEFAULTS  = {
        'width': 800,        
        'height': 600,
        'fps': 60
    }

    def __init__(self, 
            width: int = 800, 
            height: int = 600, 
            fps: int = 60):

        init_args = locals()

        for var_name, default_value in self.DEFAULTS.items():
            value = init_args.get(var_name)
            setattr(self, var_name, value if value is not None else default_value)


        pygame.init()
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Basic Pygame Loop")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.looping_circle = LoopingCircle(width, height)
        self.font = pygame.font.SysFont(None, 24)

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
        densities = self.DFS.get_normalized_densities()
        colors = [colormap(density, (0, 0, 255), (255, 255, 255), (255, 0, 0)) for density in densities]
        for point, color in zip(self.DFS.particles, colors):
            pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 3)

    def draw_fps(self):
        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (200, 200, 200))
        self.screen.blit(fps_text, (10, 10))

    def draw(self):
        # self.looping_circle.draw(self.screen)
        self.draw_particles()
        self.draw_fps()

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
    main()
