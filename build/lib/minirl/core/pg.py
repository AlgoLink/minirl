#!/usr/bin/env python3
import numpy as np
import scipy.stats
from itertools import count
from collections import deque

from ..common.optim import adam

# TODO: add weight saving and loading?

"""

This file implements the standard vanilla REINFORCE algorithm, also
known as Monte Carlo Policy Gradient.

The main neural network logic is contained in the PolicyNetwork class,
with more algorithm specific code, including action taking and loss
computing contained in the REINFORCE class.  (NOTE: this only supports discrete actions)


    Resources:
        Sutton and Barto: http://incompleteideas.net/book/the-book-2nd.html
        Karpathy blog: http://karpathy.github.io/2016/05/31/rl/


    Glossary:
        (w.r.t.) = with respect to (as in taking gradient with respect to a variable)
        (h or logits) = numerical policy preferences, or unnormalized probailities of actions
"""


class PolicyNetwork(object):
    """
    Neural network policy. Takes in observations and returns probabilities of
    taking actions.

    ARCHITECTURE:
    {affine - relu } x (L - 1) - affine - softmax

    """

    def __init__(
        self, ob_n, ac_n, hidden_dim=64, dtype=np.float32, lr=0.001, sample_len=7
    ):
        """
        Initialize a neural network to choose actions

        Inputs:
        - ob_n: Length of observation vector
        - ac_n: Number of possible actions
        - hidden_dims: List of size of hidden layer sizes
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.ob_n = ob_n
        self.ac_n = ac_n
        self.hidden_dim = H = hidden_dim
        self.dtype = dtype
        self.lr = lr

        self.sample_len = sample_len

        # Initialize all weights (model params) with "Xavier Initialization"
        # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
        # bias init = zeros()
        self.params = {}
        self.params["W1"] = (-1 + 2 * np.random.rand(ob_n, H)) / np.sqrt(ob_n)
        self.params["b1"] = np.zeros(H)
        self.params["W2"] = (-1 + 2 * np.random.rand(H, ac_n)) / np.sqrt(H)
        self.params["b2"] = np.zeros(ac_n)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

        # Neural net bookkeeping
        self.cache = {}
        self.grads = {}
        # Configuration for Adam optimization
        self.optimization_config = {"learning_rate": self.lr}
        self.adam_configs = {}
        for p in self.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.adam_configs[p] = d

    ### HELPER FUNCTIONS
    def _zero_grads(self):
        """Reset gradients to 0. This should be called during optimization steps"""
        for g in self.grads:
            self.grads[g] = np.zeros_like(self.grads[g])

    def _add_to_cache(self, name, val):
        """Helper function to add a parameter to the cache without having to do checks"""
        if name in self.cache:
            self.cache[name].append(val)
        else:
            self.cache[name] = deque(maxlen=self.sample_len)
            self.cache[name].append(val)

    def _update_grad(self, name, val):
        """Helper fucntion to set gradient without having to do checks"""
        if name in self.grads:
            self.grads[name] += val
        else:
            self.grads[name] = val

    def _softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs

    ### MAIN NEURAL NETWORK STUFF
    def forward(self, x, test):
        """
        Forward pass observations (x) through network to get probabilities
        of taking each action

        [input] --> affine --> relu --> affine --> softmax/output

        """
        p = self.params
        W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]

        # forward computations
        affine1 = x.dot(W1) + b1
        relu1 = np.maximum(0, affine1)
        affine2 = relu1.dot(W2) + b2

        logits = affine2  # layer right before softmax (i also call this h)
        # pass through a softmax to get probabilities
        probs = self._softmax(logits)

        # cache values for backward (based on what is needed for analytic gradient calc)
        if not test:
            self._add_to_cache("fwd_x", x)
            self._add_to_cache("fwd_affine1", affine1)
            self._add_to_cache("fwd_relu1", relu1)
        return probs

    def backward(self, dout):
        """
        Backwards pass of the network.

        affine <-- relu <-- affine <-- [gradient signal of softmax/output]

        Params:
            dout: gradient signal for backpropagation


        Chain rule the derivatives backward through all network computations
        to compute gradients of output probabilities w.r.t. each network weight.
        (to be used in stochastic gradient descent optimization (adam))
        """
        p = self.params
        W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]

        # get values from network forward passes (for analytic gradient computations)
        fwd_relu1 = np.concatenate(self.cache["fwd_relu1"])
        fwd_affine1 = np.concatenate(self.cache["fwd_affine1"])
        fwd_x = np.concatenate(self.cache["fwd_x"])

        # Analytic gradient of last layer for backprop
        # affine2 = W2*relu1 + b2
        # drelu1 = W2 * dout
        # dW2 = relu1 * dout
        # db2 = dout
        drelu1 = dout.dot(W2.T)
        dW2 = fwd_relu1.T.dot(dout)
        db2 = np.sum(dout, axis=0)

        # gradient of relu (non-negative for values that were above 0 in forward)
        daffine1 = np.where(fwd_affine1 > 0, drelu1, 0)

        # affine1 = W1*x + b1
        # dx
        dW1 = fwd_x.T.dot(daffine1)
        db1 = np.sum(daffine1)

        # update gradients
        self._update_grad("W1", dW1)
        self._update_grad("b1", db1)
        self._update_grad("W2", dW2)
        self._update_grad("b2", db2)

        # reset cache for next backward pass
        self.cache = {}


class PGAgent(object):
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
    """

    def __init__(self, ob_n, ac_n, gamma=0.99, score_len=7):
        ob_n = ob_n
        ac_n = ac_n
        self.gamma = gamma
        self.score_len = score_len
        self.best_score = 0
        self.policy = PolicyNetwork(ob_n, ac_n)
        # RL specific bookkeeping
        self.saved_action_gradients = []
        self.rewards = deque(maxlen=score_len)
        self.score_history = deque(maxlen=score_len)

    def act(self, obs, test=False):
        """
        Pass observations through network and sample an action to take. Keep track
        of dh to use to update weights
        """
        obs = np.reshape(obs, [1, -1])
        netout = self.policy.forward(obs, test)[0]

        probs = netout
        # randomly sample action based on probabilities
        action = np.random.choice(self.policy.ac_n, p=probs)
        # derivative that pulls in direction to make actions taken more probable
        # this will be fed backwards later
        # (see README.md for derivation)
        dh = -1 * probs
        dh[action] += 1
        self.saved_action_gradients.append(dh)

        return action

    def cal_disconted_returns(self, rewards):
        """
        Calculate discounted reward and then normalize it
        (see Sutton book for definition)
        Params:
            rewards: list of rewards for every episode
        """
        returns = np.zeros(len(rewards))

        next_return = 0  # 0 because we start at the last timestep
        for t in reversed(range(0, len(rewards))):
            next_return = rewards[t] + self.gamma * next_return
            returns[t] = next_return
        # normalize for better statistical properties
        returns = (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps
        )
        return returns

    def learn(self):
        """
        finish_episode: At the end of the episode, calculate the discounted return for each time step and update the model parameters
        """
        action_gradient = np.array(self.saved_action_gradients)
        returns = self.cal_disconted_returns(self.rewards)
        # Multiply the signal that makes actions taken more probable by the discounted
        # return of that action.  This will pull the weights in the direction that
        # makes *better* actions more probable.
        self.policy_gradient = np.zeros(action_gradient.shape)
        for t in range(0, len(returns)):
            self.policy_gradient[t] = action_gradient[t] * returns[t]

        # negate because we want gradient ascent, not descent
        self.policy.backward(-self.policy_gradient)

        # run an optimization step on all of the model parameters
        for p in self.policy.params:
            next_w, self.policy.adam_configs[p] = adam(
                self.policy.params[p],
                self.policy.grads[p],
                config=self.policy.adam_configs[p],
            )
            self.policy.params[p] = next_w
        self.policy._zero_grads()  # required every call to adam

        # reset stuff
        del self.rewards[:]
        del self.saved_action_gradients[:]

    def add_step_reward(self, reward):
        self.rewards.append(reward)
        self.score_history.append(reward)
        if len(self.score_history) >= self.score_len:
            avg_score = np.mean(self.score_history)
            self.score_history.clear()
            if avg_score > self.best_score:
                self.best_score = avg_score

    def train(self):
        avg_reward = []
        ep_reward = 0


def main():
    """Run REINFORCE algorithm to train on the environment"""
    avg_reward = []
    for i_episode in count(1):
        ep_reward = 0
        obs, _ = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = reinforce.select_action(obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            reinforce.rewards.append(reward)

            if args.render_interval != -1 and i_episode % args.render_interval == 0:
                env.render()

            if done:
                break

        reinforce.finish_episode()

        if i_episode % args.log_interval == 0:
            print("Ave reward: {}".format(sum(avg_reward) / len(avg_reward)))
            avg_reward = []

        else:
            avg_reward.append(ep_reward)
