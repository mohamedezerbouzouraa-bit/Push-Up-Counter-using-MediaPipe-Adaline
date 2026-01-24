import numpy as np

class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False
        if random_state:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = 0
            for xi, target in zip(X, y):
                cost += self._update_weights(xi, target)
            avg_cost = cost / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        for xi, target in zip(X, y):
            self._update_weights(xi, target)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(m + 1)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights."""
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi * error
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation (identity)."""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(X) >= 0.0, 1, -1)
