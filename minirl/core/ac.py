import numpy as np
from ..common.net import TwoNN
from ..common.optim import MyAdam


class ACAgent:
    def __init__(self, state_dim, n_actions, hidden_layers=64, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.n_actions = n_actions
        self.actor = TwoNN(state_dim, hidden_layers, n_actions)
        self.critic = TwoNN(state_dim, hidden_layers, 1)
        self.adam_ac = MyAdam(weights=self.actor.model, learning_rate=lr)
        self.adam_cr = MyAdam(weights=self.critic.model, learning_rate=lr)
        self.cache = {}

    def grads_manual_np(self, tP, y_train, th, X, W2, adv, N=1):
        # print('shape:', tP.shape, y_train.shape)
        grads = {}

        dz2 = (tP - y_train) * adv / N
        dw2 = dz2.T @ th
        dh = dz2 @ W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dh[th <= 0] = 0  # equal sign is extremely important.
        X_reshape = np.reshape(X, [1, -1])
        dw1 = dh.T @ X_reshape
        db1 = np.sum(dh, axis=0, keepdims=True)
        grads = {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}
        return grads

    def grads_manual_critic_np(
        self, yhat, y, th, th_next, state, next_state, w2, done, N=1
    ):
        grads = {}

        dz2 = (yhat - y) / N
        cth = (1 - done) * self.gamma * th_next - th
        dw2 = dz2.T @ cth

        dh = dz2 @ w2

        db2 = np.sum(dz2 * ((1 - done) * self.gamma - 1), axis=0, keepdims=True)

        dz1n = dh.copy()
        dz1n[th_next <= 0] = 0  # equal sign is extremely important.

        dz1 = dh.copy()
        dz1[th <= 0] = 0  # equal sign is extremely important.
        next_state_reshape = np.reshape(next_state, [1, -1])
        state_reshape = np.reshape(state, [1, -1])
        dw1 = (
            1 - done
        ) * self.gamma * dz1n.T @ next_state_reshape - dz1.T @ state_reshape

        db1 = np.sum(((1 - done) * self.gamma * dz1n - dz1), axis=0, keepdims=True)

        grads = {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}
        return grads

    def act(self, state):
        probs_np = self.actor.forward(state)
        ath_np = self.actor.h.copy()
        action = np.random.choice(self.n_actions, p=probs_np[0])
        self.actor._add_to_cache("ath_np", ath_np)
        self.actor._add_to_cache("action", action)
        self.actor._add_to_cache("probs_np", probs_np)
        return action

    def learn(self, state, reward, next_state, done=False):
        y_next = self.critic.forward(np.reshape(next_state, [1, -1]))
        th_next = self.critic.h.copy()
        y = self.critic.forward(np.reshape(state, [1, -1]))
        th = self.critic.h.copy()
        yhat = reward + (1 - done) * self.gamma * y_next

        grads_critic_np = self.grads_manual_critic_np(
            yhat, y, th, th_next, state, next_state, self.critic.model["w2"], done, N=1
        )

        self.adam_cr.update(grads_critic_np)

        # for k,v in grads_critic_np.items():
        #     critic_np.model[k] -=lr*grads_critic_np[k]

        advantage = yhat - y
        action = self.actor.cache["action"][-1]
        yt = np.eye(self.n_actions)[action].reshape(1, -1)
        probs_np = np.concatenate(self.actor.cache["probs_np"])
        ath_np = np.concatenate(self.actor.cache["ath_np"])
        grads_actor_np = self.grads_manual_np(
            probs_np, yt, ath_np, state, self.actor.model["w2"], advantage, N=1
        )

        # for k,v in grads_actor_np.items():
        #     actor_np.model[k] -=lr*grads_actor_np[k]
        self.adam_ac.update(grads_actor_np)
        self.actor.cache = {}
