import matplotlib.pyplot as plt
import numpy as np
import SVM
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap


def test_plot(X, y, svm_model):
    xlim = [np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1]
    ylim = [np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1]
    xx, yy = np.meshgrid(np.linspace(*xlim,num =500), np.linspace(*ylim, num =500))
    rgb = np.array([[210, 0, 0], [0, 0, 150]]) / 255.0

    svm_model.fit(X, y)
    z_model = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='cool')
    plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)


array, group = make_blobs(n_samples=50, centers=2, random_state=2, cluster_std=1.5)
test_plot(array, group, SVM.SVM(kernel='linear', C=10, max_iter=50))
plt.show()
