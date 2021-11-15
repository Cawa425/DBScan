from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns;
import numpy as np

import SVM

sns.set()
from sklearn.datasets import make_blobs, make_circles
from matplotlib.colors import ListedColormap


def test_plot(X, y, svm_model):
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
    xx, yy = np.meshgrid(np.linspace(*xlim, num=700), np.linspace(*ylim, num=700))
    rgb = np.array([[210, 0, 0], [0, 0, 150]]) / 255.0

    svm_model.fit(X, y)
    z_model = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)


X, y = make_blobs(n_samples=100, centers=2, random_state=1, cluster_std=1.4)
test_plot(X, y, SVM.SVM(kernel='linear', C=10, max_iter=60))

plt.show()
