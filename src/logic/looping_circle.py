


import pygame


class LoopingCircle:
    def __init__(self, width, height) -> None:
        self.size = (width, height)
        self.pos = [width // 4., height // 2.]
        self.speed = 200  # pixels per second
        self.radius = 20

    def update(self, dt: float):
        # Move circle horizontally and wrap around screen
        self.pos[0] += self.speed * dt
        if self.pos[0] - self.radius > self.size[0]:
            self.pos[0] = -self.radius

    def draw(self, screen: pygame.Surface):
        screen.fill((30, 30, 30))
        pygame.draw.circle(screen, (100, 200, 255), (int(self.pos[0]), int(self.pos[1])), self.radius)




