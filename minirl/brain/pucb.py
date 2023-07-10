import numpy as np
import copy
import pickle


# see e.g. here: https://deepnotes.io/softmax-crossentropy
def softmax(x):
    """Numerically stable softmax"""
    max_ = np.max(x, axis=-1, keepdims=True)  # shape: (n_batch, 1)
    ex = np.exp(x - max_)  # shape: (n_batch, n_out)
    ex_sum = np.sum(ex, axis=-1, keepdims=True)  # shape: (n_batch, 1)
    return ex / ex_sum


def argmax_rand(arr):
    """Pick max value, break ties randomly."""
    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 1

    # break ties randomly, np.argmax() always picks first max
    return np.random.choice(np.flatnonzero(arr == arr.max()))


class PUCB:
    """Similar to UCB, but allows injecting priors.

    Params:
        env (BanditEnv): environment to solve
        nb_total (int): how many time steps in total
        c_param (float): exploration param, higher means explore more
        priors (np.ndarray): prior probability distribution, must sum to 1

    Returns:
        Q (np.ndarray): array with value estimates, shape: [nb_arms]
        rewards (np.ndarray): history of rewards, shape: [nb_total]
    """

    def __init__(self, action_n, c_param=np.sqrt(2), model_db=None):
        self.c_param = c_param
        self.action_n = action_n
        self._model_db = model_db
        self._init_model()

    def _init_model(self):
        # V:sum_reward Q: mean_reward N:n(a)
        self.V = np.zeros(self.action_n)
        self.N = np.zeros(self.action_n)
        self.Q = np.zeros(self.action_n)

    def act(self, priors, model_id, allowed_actions=None, expose_a=True):
        assert self.c_param == float(self.c_param) and self.c_param >= 0
        assert isinstance(priors, np.ndarray)
        assert len(priors.shape) == 1
        assert abs(priors.sum() - 1) < 1e-6

        v, q, n = self.get_weights(model_id)
        self.set_weights(v, q, n)

        scores = self.Q + self.c_param * priors * np.sqrt(self.N.sum()) / (1 + self.N)
        if allowed_actions is not None:
            new_scores = [float(scores[a]) for a in allowed_actions]
            best_action = argmax_rand(np.array(new_scores))
            A = allowed_actions[best_action]
        else:
            A = argmax_rand(scores)
        if expose_a:
            self.N[A] += 1

        self.save_weights(model_id, self.V, self.Q, self.N)
        self._init_model()

        return A

    def learn(self, action, reward, model_id):
        A = action
        R = reward
        v, q, n = self.get_weights(model_id)
        self.set_weights(v, q, n)
        self.V[A] += R
        self.Q[A] = self.V[A] / self.N[A]
        self.save_weights(model_id, self.V, self.Q, self.N)
        self._init_model()

    def get_model_key(self, model_id):
        return f"{model_id}:pucb"

    def set_weights(self, v, q, n):
        self.V = copy.deepcopy(v)
        self.Q = copy.deepcopy(q)
        self.N = copy.deepcopy(n)

    def get_weights(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            v, q, n = self.V, self.Q, self.N

        else:
            v, q, n = pickle.loads(_model)

        return v, q, n

    def save_weights(self, model_id, v, q, n):
        model_key = self.get_model_key(model_id)
        self._model_db.set(model_key, pickle.dumps([v, q, n]))
