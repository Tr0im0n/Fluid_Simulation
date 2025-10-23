import sys
import pygame

from looping_circle import LoopingCircle

class Game:

    DEFAULTS  = {
        'width': 800,        
        'height': 600,
        'fps': 60
    }

    def __init__(self, width: int = 800, height: int = 600, fps: int = 60):

        init_args = locals()

        for var_name, default_value in self.DEFAULTS.items():
            value = init_args.get(var_name)
            setattr(self, var_name, value if value is not None else default_value)


        pygame.init()
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Basic Pygame Loop")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.running = True
        
        self.circle = LoopingCircle(width, height)
        self.font = pygame.font.SysFont(None, 24)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def draw_fps(self):
        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (200, 200, 200))
        self.screen.blit(fps_text, (10, 10))

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(self.fps) / 1000.0  # seconds since last frame
                self.handle_events()
                self.circle.update(dt)
                self.circle.draw(self.screen)
                self.draw_fps()
                pygame.display.flip()
        finally:
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    Game().run()