import numpy as np
from .reinfore_baseline_config import agent_parameters

ASSERTS = True


class TabularStateValueFunction:
    """Tabular state-value function 'approximator'"""

    def __init__(self, lr, nb_states):
        self._lr = lr
        self._w = np.zeros(nb_states)

    def evaluate(self, state):
        # or onehot(state, nb_states) @ self._w
        return self._w[state]

    def train(self, state, target):
        value = self.evaluate(state)
        self._w[state] += self._lr * (target - value)


class TabularStateActionValueFunction:
    """Tabular state-action-value function 'approximator'"""

    def __init__(self, lr, nb_states, nb_actions):
        self._lr = lr
        self._q = np.zeros((nb_states, nb_actions))

    def evaluate(self, state, action):
        return self._q[state, action]

    def train(self, state, action, target):
        value = self.evaluate(state, action)
        self._q[state, action] += self._lr * (target - value)


def softmax(x):
    """Numerically stable softmax"""
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


class TabularSoftmaxPolicy:
    """Tabular action-state function 'approximator'"""

    def __init__(self, lr, nb_states, nb_actions, init_theta=None):
        self._lr = lr  # learning rate
        self.n_act = nb_actions
        self._theta = np.zeros((nb_states, nb_actions))  # weights
        if init_theta is not None:
            assert init_theta.dtype == np.float64
            assert init_theta.shape == self._theta.shape
            self._theta = init_theta

    def pi(self, state):
        """Return policy, i.e. probability distribution over actions."""
        h_vec = self._theta[state]
        prob_vec = softmax(h_vec)  # shape=[n_act], e.q. 13.2
        if ASSERTS:
            assert prob_vec[0] != 0.0 and prob_vec[0] != 1.0
            assert prob_vec[1] != 0.0 and prob_vec[1] != 1.0
        return prob_vec

    def update(self, state, action, disc_return):
        x_s = np.zeros(self.n_act)
        x_s[action] = 1  # feature vector, one-hot
        prob = self.pi(state)
        grad_s = x_s - prob
        self._theta[state] += self._lr * disc_return * grad_s


class ReinforceBaseline:
    """REINFORCE algorithm.

    Params:
        env: OpenAI-like environment
        ep (int): number of episodes to run
        gamma (float): discount factor
        alpha_w (float): learning rate for state-value function
        alpha_theta (float): learning rate for policy
        init_theta (np.array): initialize policy weights, default np.zeros()
    """

    def __init__(self, config=agent_parameters, init_theta=None):
        self.config = config
        self.init_theta = init_theta
        self._init_model(config=config)

    def _init_model(self, config):
        self.alpha_w = config["alpha_w"]
        self.nb_states = config["obs_dim"]
        self.alpha_q = config["alpha_q"]
        self.nb_actions = config["action_n"]
        self.alpha_theta = config["alpha_theta"]
        self.gamma = config["gamma"]

        self.v_hat = TabularStateValueFunction(
            lr=self.alpha_w, nb_states=self.nb_states
        )
        self.q_hat = TabularStateActionValueFunction(
            lr=self.alpha_q, nb_states=self.nb_states, nb_actions=self.nb_actions
        )

        self.policy = TabularSoftmaxPolicy(
            lr=self.alpha_theta,
            nb_states=self.nb_states,
            nb_actions=self.nb_actions,
            init_theta=self.init_theta,
        )

    def act(self, state):

        return np.random.choice(range(self.nb_actions), p=self.policy.pi(state))

    def learn(self, traj, baseline=True, target="Gt"):
        T = len(traj) - 1

        for t in range(0, T):
            St, Rt, _, At = traj[t]  # (st, rew, done, act)
            Gt = sum(
                [self.gamma ** (k - t - 1) * traj[k][1] for k in range(t + 1, T + 1)]
            )
            self.v_hat.train(St, Gt)  # delta calculated internally
            self.q_hat.train(St, At, Gt)
            if target == "Gt":
                tar = Gt
            elif target == "Qt":
                tar = self.q_hat.evaluate(St, At)
            else:
                raise ValueError("Unknown target:", target)

            delta = tar - self.v_hat.evaluate(St) if baseline else tar
            self.policy.update(St, At, self.gamma**t * delta)
