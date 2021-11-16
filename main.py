import random
import pygame
import numpy as np
import ByasAlg

#point colors
number_of_colors = 10  # макс цветов
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(number_of_colors)]


def init_pygame():
    global WIDTH
    WIDTH = 1000
    global HEIGHT
    HEIGHT = 600
    FPS = 60
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Game")
    screen.fill((255, 255, 255))
    pygame.time.Clock().tick(FPS)

    return screen


class Point(object):
    def __init__(self, x, y, group):
        self.x = x
        self.y = y
        self.group = group

    def color(self):
        return colors[self.group]


def generate_start_points(count, max_classes):
    points = []
    for cluster in range(count):
        new_x = random.randrange(0, WIDTH)
        new_y = random.randrange(0, HEIGHT)
        group = random.randrange(0, max_classes)
        new_point = Point(new_x, new_y, group)
        points.append(new_point)
    return points


def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def draw_points(points, screen):
    for point in points:
        pygame.draw.circle(screen, point.color(), (point.x, point.y), 3)


screen = init_pygame()
running = True

start_data = generate_start_points(200, 4)
data = [(point.x, point.y, point.group) for point in start_data]
model = ByasAlg.make_model(data)

draw_points(start_data, screen)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left click
                point = Point(event.pos[0], event.pos[1], 0)
                var = [point.x, point.y]
                label = ByasAlg.predict(model, var)
                print('Data=%s, Predicted: %s' % (var, label))
                point.group = label
                pygame.draw.circle(screen, point.color(), (point.x, point.y), 3)
                start_data.append(point)
                draw_points(start_data, screen)
    pygame.display.update()
