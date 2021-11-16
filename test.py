import random

import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.svm._libsvm import predict

import SVM
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap



def test_plot(X, y, svm_model):
    global xlim
    global ylim
    xlim = [np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1]
    ylim = [np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1]
    xx, yy = np.meshgrid(np.linspace(*xlim, num=500), np.linspace(*ylim, num=500))
    rgb = np.array([[210, 0, 0], [0, 0, 150]]) / 255.0

    svm_model.fit(X, y)
    z_model = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='cool')
    plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)


array, group = make_blobs(n_samples=10, centers=2, random_state=2, cluster_std=1.5)
model  =SVM.SVM(c=10, max_iter=50)
test_plot(array, group, model)
plt.show()


points_to_add = 10
for _ in range(points_to_add):
    itemX = random.uniform(xlim[0], xlim[1])
    itemY = random.uniform(ylim[0], ylim[1])
    item = [[itemX, itemY]]
    array = numpy.append(array, item, axis=0)
    group = model.predict(array)
    test_plot(array, group, model)
    plt.show()

