import random
import pygame
import numpy as np
from random import randrange
import collections



class Point(object):
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.group = 0


def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# point colors
number_of_colors = 10  # макс цветов
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
    global WIDTH
    WIDTH = 1000
    global HEIGHT
    HEIGHT = 600
    FPS = 60
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Game")
    screen.fill(WHITE)
    clock.tick(FPS)
    return screen


def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_color(point):
    return colors[point.group]


def get_color_by_index(index):
    return colors[index]


def generate_start_clusters(clusters_count):
    points = []

    for cluster in range(clusters_count):
        screen_offset = 50
        points_on_cluster = random.randint(1, 200)
        center = Point(random.randint(screen_offset, WIDTH - screen_offset),
                       random.randint(screen_offset, HEIGHT - screen_offset), 1)
        for _ in range(points_on_cluster):
            point_offset = random.randint(1, 100)
            new_x = random.randrange(center.x - point_offset, center.x + point_offset, 1)
            new_y = random.randrange(center.y - point_offset, center.y + point_offset, 1)
            new_point = Point(new_x, new_y, get_color_by_index(cluster))
            new_point.group = cluster
            points.append(new_point)

    return points


def create_new_point(point):
    return Point(point.x, point.y, point.color)



def draw_points(points, screen):
    for point in points:
        draw_point(point, screen)


def draw_point(point, screen):
    point.color = get_color(point)
    pygame.draw.circle(screen, point.color, (point.x, point.y), 3)


def knn_start(points, new_point, k):
    dicti = {}
    for point in points:
        dist = get_dist(point, new_point)
        group = point.group
        dicti[dist] = group
    dicti = collections.OrderedDict(sorted(dicti.items()))
    dicti = {key: val for key, val in dicti.items() if val != -1}
    result = list(dicti.values())[:k]
    group = collections.Counter(result).most_common()[0][0]
    new_point.group = group
    new_point.color = get_color(new_point)
    pass


def cross_validation_split(dataset, folds=10):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


if __name__ == '__main__':
    points = []
    clock = pygame.time.Clock()
    screen = init_pygame()
    running = True

    points += generate_start_clusters(5)
    draw_points(points, screen)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    point = create_new_point(Point(event.pos[0], event.pos[1], BLACK))
                    pygame.draw.circle(screen, point.color, (point.x, point.y), 3)
                    knn_start(points, point, 3)
                    points.append(point)
                    draw_points(points, screen)
        pygame.display.update()
