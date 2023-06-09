import numpy as np
import pickle
import copy  # for deepcopy of model parameters
from .one_ac_config import agent_parameters
import traceback


def softmax(x):
    """Numerically stable Softmax over last dimension"""
    max_ = np.max(x, axis=-1, keepdims=True)  # shape: (..., 1)
    ex = np.exp(x - max_)  # shape: (..., nb_act)
    ex_sum = np.sum(ex, axis=-1, keepdims=True)  # shape: (..., 1)
    return ex / ex_sum  # shape: (..., nb_act)


class TabularSoftmaxPolicy:
    """Tabular action-state function 'approximator'"""

    def __init__(self, lr, nb_states, nb_actions, init_theta=None):
        assert isinstance(lr, float)
        assert isinstance(nb_states, tuple)
        assert isinstance(nb_actions, int)
        self._lr = lr  # learning rate
        self.n_act = nb_actions
        self._theta = np.zeros((*nb_states, nb_actions))  # weights
        if init_theta is not None:
            assert init_theta.dtype == np.float64
            assert init_theta.shape == self._theta.shape
            self._theta = init_theta

    def pi(self, state):
        """Return policy, i.e. probability distribution over actions."""
        assert isinstance(state, (int, tuple))
        assert self._theta.ndim == 2 if isinstance(state, int) else len(state) + 1

        h_vec = self._theta[state]
        prob_vec = softmax(h_vec)  # shape=[n_act], e.q. 13.2
        assert prob_vec[0] != 0.0 and prob_vec[0] != 1.0
        assert prob_vec[1] != 0.0 and prob_vec[1] != 1.0

        assert prob_vec.ndim == 1
        return prob_vec

    def pi_all(self):
        return softmax(self._theta)

    def update(self, state, action, disc_return):
        x_s = np.zeros(self.n_act)
        x_s[action] = 1  # feature vector, one-hot
        prob = self.pi(state)
        grad_s = x_s - prob
        self._theta[state] += self._lr * disc_return * grad_s


class TabularStateValueFunction:
    """Tabular state-value function 'approximator'"""

    def __init__(self, lr, nb_states):
        assert isinstance(lr, float)
        assert isinstance(nb_states, tuple)
        self._lr = lr
        self._w = np.zeros(nb_states)

    def evaluate(self, state):
        assert isinstance(state, (int, tuple))
        assert self._w.ndim == 1 if isinstance(state, int) else len(state)
        # or onehot(state, nb_states) @ self._w
        return self._w[state]

    def evaluate_all(self):
        return self._w.copy()

    def train(self, state, target):
        assert isinstance(state, (int, tuple))
        assert self._w.ndim == 1 if isinstance(state, int) else len(state)
        assert isinstance(target, float)
        value = self.evaluate(state)
        self._w[state] += self._lr * (target - value)


class OneAC:
    """Sarsa (on-policy TD control)

    Params:
        env - environment
        ep - number of episodes to run
        gamma - discount factor [0..1]
        alpha_w (float): learning rate for state-value function
        alpha_theta (float): learning rate for policy
    """

    def __init__(self, config=agent_parameters):
        self.name = "one_step_actor_critic"
        self.config = config
        self._init_model(config)

    def _init_model(self, config):
        self.gamma = config["gamma"]
        self.act_n = config["action_n"]
        self.obs_dim = config["obs_dim"]
        self._model_db = config["model_db"]
        self.alpha_w = config["alpha_w"]
        self.alpha_theta = config["alpha_theta"]

        self.v_hat = TabularStateValueFunction(lr=self.alpha_w, nb_states=self.obs_dim)
        self.policy = TabularSoftmaxPolicy(
            lr=self.alpha_theta, nb_states=self.obs_dim, nb_actions=self.act_n
        )
        self.I = 1

    def act(self, st, model_id):
        w, I, theta = self.load_weights(model_id)
        self.set_weights(w, I, theta)
        action = np.random.choice(range(self.act_n), p=self.policy.pi(st))
        self._init_model(self.config)
        return action

    def learn(self, state, action, reward, next_state, model_id, done=False):
        w, I, theta = self.load_weights(model_id)
        self.set_weights(w, I, theta)

        R = reward
        S_ = next_state
        S = state
        A = action
        target = R + self.gamma * self.v_hat.evaluate(S_) if not done else R
        delta = target - self.v_hat.evaluate(S)
        self.v_hat.train(S, target)
        self.policy.update(S, A, self.I * delta)
        self.I *= self.gamma
        _w = self.v_hat._w
        _I = self.I
        _theta = self.policy._theta
        self.save_weights(model_id, _w, _I, _theta)
        self._init_model(self.config)

    def get_weights(self, model_id):
        # return self.weights, self.biases
        w, I, theta = self.load_weights(model_id)
        return w, I, theta

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, w, I, theta):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.v_hat._w = copy.deepcopy(w)
        self.I = copy.deepcopy(I)
        self.policy._theta = copy.deepcopy(theta)

    def model_params_key(self, model_id):
        return f"{model_id}:oneac"

    def save_weights(self, model_id, w, I, theta):
        if self._model_db is None:
            pickle.dump(
                [w, I, theta],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.model_params_key(model_id)
            self._model_db.set(model_key, pickle.dumps([w, I, theta]))

    def load_weights(self, model_id=None):
        try:
            if self._model_db is None:
                _w, _I, _theta = pickle.load(open("{}.pickle".format(model_id), "rb"))
            else:
                model_key = self.model_params_key(model_id)
                model = self._model_db.get(model_key)
                if model is not None:
                    _w, _I, _theta = pickle.loads(model)
                    return _w, _I, _theta
                else:
                    return self.v_hat._w, self.I, self.policy._theta

        except:
            print("Could not load weights: File Not Found, use default")
            print(traceback.format_exc())

            return self.v_hat._w, self.I, self.policy._theta
