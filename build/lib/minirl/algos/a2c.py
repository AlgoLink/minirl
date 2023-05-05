from ..models import activation, loss, optim
from ..models.nn import NeuralNetwork
from ..agents.base import Actor, Critic
import numpy as np
import scipy.signal


class A2C:
    def __init__(
        self,
        obs_n=None,
        ac_n=None,
        hidden_layers=[32],
        lr=1e-4,
        lr_c=1e-3,
        gamma=0.99,
        batch_size=7,
        model_db=None,
    ):
        self.lr_policy = lr
        self.lr_value = lr_c
        self.num_actions = ac_n
        self.num_states = obs_n
        self.gamma = gamma
        self._model_db = model_db

        self.actor = Actor(
            obs_n,
            ac_n,
            env_action_limit=(-1, 1),
            learning_rate=1e-4,
            hidden_layers=hidden_layers,
            model_db=self._model_db,
            wgt_name="actor",
            batch_size=batch_size,
        )

        self.critic = Critic(
            obs_n,
            ac_n,
            learning_rate=1e-3,
            hidden_layers=hidden_layers,
            model_db=self._model_db,
            wgt_name="critic",
            batch_size=batch_size,
        )

    def act(self, state, target=False):
        action_prob = np.squeeze(self.actor.predict(state, target=target))
        # print(action_prob,"ff")

        action = np.random.choice(self.num_actions, p=action_prob)

        return action, action_prob

    def learn(
        self, states, actions, action_probs, next_states, rewards, dones, target=False
    ):
        mb_states = np.hstack(states)
        mb_next_states = np.hstack(next_states)

        values = self.critic.predict(mb_states, target)
        next_values = self.critic.predict(mb_next_states, target)

        init_gradient = loss.cross_entropy_loss_deriv(
            np.array(actions), np.vstack(action_probs).T
        )
        drs = np.vstack(rewards)
        mb_clipped_rewards = self._discount(
            rewards, self.gamma
        )  # self.discount_rewards(drs, self.gamma)
        mb_targets = np.array(mb_clipped_rewards) + self.gamma * next_values  # * (
        # 1 - np.array(dones)
        # )
        td_errors = mb_targets - values

        init_gradient = init_gradient * td_errors

        self.critic.train(mb_states, mb_targets)
        self.actor.train(mb_states, actions, action_gradient=init_gradient)

    def discount_rewards(self, rewards, gamma):
        """take 1D float array of rewards and compute discounted reward"""
        r = rewards
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = (
                    0  # reset the sum, since this was a game boundary (pong specific!)
                )
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add

        # discounted_r -= np.mean(discounted_r)
        # discounted_r /= np.std(discounted_r)
        return discounted_r

    def _discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
