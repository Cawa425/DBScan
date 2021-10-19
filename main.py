import random
import pygame
import numpy as np
from pygame.time import wait


class Point(object):
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.group = 0
        self.is_border = False


def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# point colors
number_of_colors = 8  # макс цветов
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(number_of_colors)]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

radius_random = 10
count_neighbours_for_group = 4
radius_for_group = 30
current_group_number = 0


def init_pygame():
    WIDTH = 480  # ширина игрового окна
    HEIGHT = 600  # высота игрового окна
    FPS = 60
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Game")
    screen.fill(WHITE)
    clock.tick(FPS)
    return screen


def generate_points(point):
    count_points = random.randint(1, 5)
    points = []
    for i in range(count_points):
        new_x = random.randrange(point.x - radius_random + 3, point.x + radius_random + 3, 1)
        new_y = random.randrange(point.y - radius_random + 3, point.y + radius_random + 3, 1)
        points.append(Point(new_x, new_y, point.color))
    return points


def get_neighbours(point, points):
    neighbours = []
    for p in points:
        if get_dist(point, p) <= radius_for_group and p.x != point.x and p.y != point.y:
            neighbours.append(p)
    return neighbours


def get_group(points):
    for point in points:
        if point.group > 0:
            return point.group
    global current_group_number
    current_group_number += 1
    return current_group_number


def is_border_point(points):
    for point in points:
        if point.group > 0:
            return True
    return False


def get_color(point):
    if point.is_border:
        return YELLOW
    if point.group == -1:
        return RED
    if point.is_border != True and point.group != -1:
        return colors[point.group]


def sort_points(points):
    zero_point = Point(0, 0, WHITE)
    for i in range(len(points) - 1):
        for j in range(len(points)):
            if get_dist(zero_point, points[i]) < get_dist(zero_point, points[j]):
                a = points[j]
                points[j] = points[i]
                points[i] = a
    return points


def get_nearest(point, points):
    nearest_point = points[0]
    for p in points:
        if get_dist(nearest_point, point) > get_dist(p, point) and p.x != point.x and p.y != point.y and p.group > 0:
            nearest_point = p
    return nearest_point


def draw_points(points, screen):
    for point in points:
        draw_point(point, screen)


def draw_point(point, screen):
    point.color = get_color(point)
    pygame.draw.circle(screen, point.color, (point.x, point.y), 3)


def resolve_border_points(points):
    for point in points:
        if point.is_border:
            nearest_point = get_nearest(point, points)
            point.group = nearest_point.group
            point.is_border = False
    return points


def clustering(points, screen):
    for point in points:
        neighbours = get_neighbours(point, points)
        if len(neighbours) >= count_neighbours_for_group - 1:
            point.group = get_group(neighbours)
        else:
            if is_border_point(neighbours):
                point.is_border = True
                point.group = get_group(neighbours)
            else:
                point.group = -1
        draw_point(point, screen)
        pygame.display.update()
        wait(150)


if __name__ == '__main__':
    points = []
    clock = pygame.time.Clock()
    screen = init_pygame()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    new_points = generate_points(Point(event.pos[0], event.pos[1], BLACK))
                    for point in new_points:
                        pygame.draw.circle(screen, point.color, (point.x, point.y), 3)
                    points += new_points
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # right click
                    points = sort_points(points)
                    clustering(points, screen)
                    points = resolve_border_points(points)
                    draw_points(points, screen)
        pygame.display.update()
