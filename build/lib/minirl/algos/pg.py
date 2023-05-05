import minirl.neural_nets.numpy as np
import numpy

from minirl.neural_nets.nn.model import ModelBase


class PolicyNetwork(ModelBase):
    def __init__(
        self, input_size, act_space, hidden_size=64, gamma=0.99
    ):  # Reward discounting factor
        super(PolicyNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.add_param("w1", (hidden_size, input_size))
        self.add_param("w2", (act_space, hidden_size))

    def forward(self, X):
        """Forward pass to obtain the action probabilities for each observation in `X`."""
        a = np.dot(self.params["w1"], X.T)
        h = np.maximum(0, a)
        logits = np.dot(h.T, self.params["w2"].T)
        # p = 1.0 / (1.0 + np.exp(-logits))
        ps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        ps /= np.sum(ps, axis=1, keepdims=True)
        return ps

    def choose_action(self, p):
        """Return an action `a` and corresponding label `y` using the probability float `p`."""
        a = 2 if numpy.random.uniform() < p else 3
        y = 1 if a == 2 else 0
        return a, y

    def loss(self, ps, ys, rs):
        # Prevent log of zero.
        ps = np.maximum(1.0e-5, np.minimum(1.0 - 1e-5, ps))
        step_losses = ys * np.log(ps) + (1.0 - ys) * np.log(1.0 - ps)
        return -np.sum(step_losses * rs)

    def discount_rewards(self, rs):
        drs = np.zeros_like(rs).asnumpy()
        s = 0
        for t in reversed(range(0, len(rs))):
            # Reset the running sum at a game boundary.
            if rs[t] != 0:
                s = 0
            s = s * self.gamma + rs[t]
            drs[t] = s
        drs -= np.mean(drs)
        drs /= np.std(drs)
        return drs


class PG:
    def __init__(self, obs_n, act_n, hidden_size=64, gamma=0.99, **kwargs):
        self.model = PolicyNetwork(obs_n, act_n, hidden_size, gamma)

        self.update_every = kwargs.pop("update_every", 10)
        self.save_every = kwargs.pop("save_every", 10)
        self.save_dir = kwargs.pop("save_dir", "./")
        self.resume_from = kwargs.pop("resume_from", None)

    def act(self, state):
        step_ps = self.model.forward(state)
        us = numpy.random.uniform(size=step_ps.shape[0])[:, np.newaxis]
        as_ = (numpy.cumsum(step_ps.asnumpy(), axis=1) > us).argmax(axis=1)

        return as_

    def learn(self, states, rewards):
        xs = np.vstack(states)
        rs = np.expand_dims(self.model.discount_rewards(rewards), axis=1)

        # Performs a forward pass and computes loss using an entire episode's data.
        def loss_func(*params):
            ps = self.model.forward(xs)
            return self.model.loss(ps, ys, rs)
