from random import randrange
from math import sqrt
from math import exp
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import itertools
import operator


def calculate_probability(x, mean, std):
    exponent = exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (sqrt(2 * pi) * std)) * exponent


def calculate_class_probabilities(summaries, point):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, std, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(point[i], mean, std)
    return probabilities


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


def make_model(dataset):
    separated_by_class = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated_by_class.items():
        sum = [(np.mean(column), np.std(column), len(column)) for column in zip(*rows)]
        del (sum[-1])
        summaries[class_value] = sum
    return summaries
