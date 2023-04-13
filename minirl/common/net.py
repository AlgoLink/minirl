import numpy as np
from collections import deque


class TwoNN:
    # numpy model
    def __init__(self, n_feature, n_hidden, n_output):
        print("simple two layer neural network")
        print(f"creating nn: #input:{n_feature} #hidden:{n_hidden} #output:{n_output}")
        self.n_output = n_output
        self.model = {}
        self.model["w1"] = np.random.randn(n_hidden, n_feature) / np.sqrt(
            n_hidden
        )  # "Xavier" initialization
        self.model["b1"] = np.zeros((1, n_hidden))
        self.model["w2"] = np.random.randn(n_output, n_hidden) / np.sqrt(
            n_output
        )  # "Xavier" initialization
        self.model["b2"] = np.zeros((1, n_output))
        self.cache = {}

    def softmax(self, Z):
        expz = np.exp(Z - np.max(Z))  # to prevent overflow
        return expz / expz.sum(axis=1, keepdims=True)  # reduce columns for each row.

    def _add_to_cache(self, name, val):
        """Helper function to add a parameter to the cache without having to do checks"""
        self.cache[name] = deque(maxlen=1)
        self.cache[name].append(val)

    def forward(self, X):
        """
        X: Nxn_feature
        """
        self.X = X
        self.h = X @ self.model["w1"].T + self.model["b1"]  # (NxD)@(DxH)
        self.h[self.h < 0] = 0  # relu
        z = self.h @ self.model["w2"].T + self.model["b2"]
        if self.n_output == 1:  # regression.
            self.out = z
            return self.out

        self.out = self.softmax(z)  # (NxH)@(HxO)
        return self.out
