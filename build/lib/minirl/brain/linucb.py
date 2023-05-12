import numpy as np
import pickle
import copy  # for deepcopy of model parameters


class LinUCB:
    def __init__(self, feature_dim, alpha=0.1, model_db=None):
        self._model_db = model_db
        self.alpha = alpha
        H = 1
        # H:hidden_dim
        theta_init = (-1 + 2 * np.random.rand(feature_dim, H)) / np.sqrt(feature_dim)
        self.A = np.eye(feature_dim)
        self.b = np.zeros((feature_dim, 1))
        # self.theta = np.zeros((feature_dim, 1))
        self.theta = theta_init

    def learn(self, x, r, model_id):
        self.set_weights(*self.get_weights(model_id))
        self.A += np.outer(x, x)
        self.b += r * x.reshape(-1, 1)
        self.theta = np.dot(np.linalg.inv(self.A), self.b)

        self.save_weights(model_id)

    def get_score(self, x, model_id):
        self.set_weights(*self.get_weights(model_id))
        beta = np.sqrt(
            self.alpha * np.log(np.linalg.det(self.A)) / np.linalg.det(self.A)
        )
        return np.dot(self.theta.T, x) + beta

    def act(self, arms, model_id):
        scores = [self.get_score(x, model_id) for x in arms]
        return np.argmax(scores)

    def get_model_key(self, model_id):
        return f"{model_id}:params"

    def get_weights(self, model_id):
        # return self.weights, self.biases
        A, b, theta = self.load_weights(model_id)
        return A, b, theta

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, A, b, theta):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.A = copy.deepcopy(A)
        self.b = copy.deepcopy(b)
        self.theta = copy.deepcopy(theta)

    def save_weights(self, model_id):
        if self._model_db is None:
            pickle.dump(
                [self.A, self.b, self.theta],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.get_model_key(model_id)
            self._model_db.set(model_key, pickle.dumps([self.A, self.b, self.theta]))

    def load_weights(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            A, b, theta = self.A, self.b, self.theta
        else:
            A, b, theta = pickle.loads(_model)
        return A, b, theta
