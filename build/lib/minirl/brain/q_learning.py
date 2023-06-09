import numpy as np
import copy  # for deepcopy of model parameters
import pickle
from collections import defaultdict
import traceback


def argmax_rand(arr):
    # break ties randomly, np.argmax() always picks first max
    return np.random.choice(np.flatnonzero(arr == np.max(arr)))


class Qlearning:
    """Q-Learning (off-policy TD control)

    Params:
        env - environment
        ep - number of episodes to run
        gamma - discount factor [0..1]
        alpha - step size (0..1]
        eps - epsilon-greedy param
    """

    def __init__(self, config):
        self.config = config
        self._init_model(config=config)

    def _init_model(self, config):
        self.gamma = config.get("gamma", 0.99)
        self.action_n = config.get("action_n", 0.99)
        self.actions = config.get("actions",["up","nochange","lower"])
        self.eps = config.get("eps", 0.2)
        self.alpha = config.get("alpha", 0.1)
        self._model_db = config.get("model_db", None)
        self.Q = defaultdict(float)  # default zero for all, terminal MUST be zero

    @property
    def params(self):
        """A dictionary containing the current policy parameters"""
        return {"Q": self.Q}

    @property
    def hyperparams(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "QLearning",
            "alpha": self.alpha,
            "actions":self.actions,
            "eps": self.eps,
            "gamma": self.gamma,
            "action_n": self.action_n,
            "model_db": self._model_db,
            "config": self.config,
        }

    def act(self, state, model_id):
        if np.random.rand() > self.eps:
            model = self.load_weights(model_id=model_id)
            self.set_weights(model)
            _action = argmax_rand([self.Q[state, a] for a in self.actions])
            action=self.actions[_action]
            self._init_model(self.config)
            return action
        else:
            return np.random.choice(self.actions)

    def learn(self, state, action, reward, next_state, model_id=None, done=False):
        model = self.load_weights(model_id=model_id)
        self.set_weights(model)
        S_, R, done = next_state, reward, done
        S = state
        A = action
        max_Q = np.max([self.Q[S_, a] for a in self.actions])
        self.Q[S, A] = self.Q[S, A] + self.alpha * (
            R + self.gamma * max_Q - self.Q[S, A]
        )
        self.save_weights(model_id=model_id, Q=self.Q)
        self._init_model(self.config)

    def get_weights(self, model_id):
        # return self.weights, self.biases
        Q = self.load_weights(model_id)
        return Q

    def set_weights(self, Q):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.Q = copy.deepcopy(Q)

    def model_params_key(self, model_id):
        params_id = self.hyperparams.get("id")
        return f"{model_id}:{params_id}"

    def save_weights(self, model_id, Q):
        if self._model_db is None:
            pickle.dump(
                Q,
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.model_params_key(model_id)
            self._model_db.set(model_key, pickle.dumps(Q))

    def load_weights(self, model_id=None):
        try:
            if self._model_db is None:
                Q = pickle.load(open("{}.pickle".format(model_id), "rb"))
            else:
                model_key = self.model_params_key(model_id)
                _Q = self._model_db.get(model_key)
                if _Q is not None:
                    Q = pickle.loads(_Q)
                    return Q
                else:
                    return self.Q

        except:
            print("Could not load weights: File Not Found, use default")
            print(traceback.format_exc())

            return self.Q
