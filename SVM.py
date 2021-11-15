import numpy as np


class SVM:
    def __init__(self, kernel='linear', C=10000.0, max_iter=100000, degree=3, gamma=1):
        self.kernel = {'poly': lambda x, y: np.dot(x, y.T) ** degree,
                       'rbf': lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)),
                       'linear': lambda x, y: np.dot(x, y.T)}[kernel]
        self.C = C
        self.max_iter = max_iter

    # ограничение параметра t, чтобы новые лямбды не покидали границ квадрата
    def restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t * u, 0, self.C) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.C) - v0)[0] / u[0]

    def fit(self, X, y):
        self.X = X.copy()
        # преобразование классов 0,1 в -1,+1; для лучшей совместимости с sklearn
        self.y = y * 2 - 1
        self.lambdas = np.zeros_like(self.y, dtype=float)
        # формула (3)
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        # выполняем self.max_iter итераций
        for _ in range(self.max_iter):
            # проходим по всем лямбда
            for idxM in range(len(self.lambdas)):
                # idxL выбираем случайно
                idxL = np.random.randint(0, len(self.lambdas))
                # формула (4с)
                Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                # формула (4a)
                v0 = self.lambdas[[idxM, idxL]]
                # формула (4b)
                k0 = 1 - np.sum(self.lambdas * self.K[[idxM, idxL]], axis=1)
                # формула (4d)
                u = np.array([-self.y[idxL], self.y[idxM]])
                # регуляризированная формула (5), регуляризация только для idxM = idxL
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                self.lambdas[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)

        # найти индексы опорных векторов
        idx, = np.nonzero(self.lambdas > 1E-15)
        # формула (1)
        self.b = np.mean((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx])

    def decision_function(self, X):
        return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b

    def predict(self, X):
        # преобразование классов -1,+1 в 0,1; для лучшей совместимости с sklearn
        return (np.sign(self.decision_function(X)) + 1) // 2