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

import numpy as np
from itertools import count
from ..common.optim import adam
from ..common.replay_buffer import ReplayMemory
import pickle

# 多步剧本有效


class PolicyNetwork(object):
    """
    Neural network policy. Takes in observations and returns probabilities of
    taking actions.

    ARCHITECTURE:
    {affine - relu } x (L - 1) - affine - softmax

    """

    def __init__(
        self,
        ob_n,
        ac_n,
        lr=1e-3,
        hidden_dim=64,
        capacity=20,
        batch_size=7,
        dtype=np.float32,
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

        self.batch_size = batch_size
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
        self.optimization_config = {"learning_rate": lr}
        self.adam_configs = {}
        for p in self.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.adam_configs[p] = d

        self.capacity = capacity
        self.replay_buffer = ReplayMemory(capacity=capacity)
        self.temp_replay_buffer = ReplayMemory(capacity=1)

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
            self.cache[name] = [val]

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
    def forward(self, x):
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
        self._add_to_cache("fwd_x", x)
        self._add_to_cache("fwd_affine1", affine1)
        self._add_to_cache("fwd_relu1", relu1)
        self.temp_replay_buffer.push([x, affine1, relu1])
        return probs

    def backward(self, dout, fwd_relu1, fwd_affine1, fwd_x):
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
        # fwd_relu1 = np.concatenate(self.cache["fwd_relu1"])
        # fwd_affine1 = np.concatenate(self.cache["fwd_affine1"])
        # fwd_x = np.concatenate(self.cache["fwd_x"])

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
        self.replay_buffer = ReplayMemory(capacity=self.capacity)


class REINFORCE(object):
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        gamma=0.99,
        model_db=None,
        grad_db=None,
        capacity=100000000,
        batch_size=9,
    ):
        ob_n = state_dim
        ac_n = act_dim
        self._model_db = model_db
        self._grad_db = grad_db

        self.policy = PolicyNetwork(
            ob_n, ac_n, capacity=capacity, batch_size=batch_size
        )
        # RL specific bookkeeping
        self.saved_action_gradients = []
        self.rewards = []
        self.gamma = gamma

    def act(self, state, model_id):
        """
        Pass observations through network and sample an action to take. Keep track
        of dh to use to update weights
        """
        obs = np.reshape(state, [1, -1])
        netout = self.policy.forward(obs)[0]

        probs = netout
        # randomly sample action based on probabilities
        action = np.random.choice(self.policy.ac_n, p=probs)
        # derivative that pulls in direction to make actions taken more probable
        # this will be fed backwards later
        # (see README.md for derivation)
        dh = -1 * probs
        dh[action] += 1
        # self.saved_action_gradients.append(dh)
        grad_key = f"{model_id}:agrad"
        _action_grad = self._grad_db.get(grad_key)
        if _action_grad is None:
            action_grad = self.saved_action_gradients
        else:
            action_grad = pickle.loads(_action_grad)
        action_grad.append(dh)
        self._grad_db.set(grad_key, pickle.dumps(action_grad))

        return action

    def calculate_discounted_returns(self, rewards):
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

    def learn(self, reward, model_id=None, learn=True):
        """
        At the end of the episode, calculate the discounted return for each time step and update the model parameters
        """
        states, affine1s, relu1s, rewards = self.update_buffer(reward, learn)

        if len(states) > 0:
            # print(len(states),"samples")
            fwd_relu1 = np.concatenate(relu1s)
            fwd_affine1 = np.concatenate(affine1s)
            fwd_x = np.concatenate(states)

            # load temp grads
            grad_key = f"{model_id}:agrad"
            _action_grad = self._grad_db.get(grad_key)
            if _action_grad is None:
                action_grad = self.saved_action_gradients
            else:
                action_grad = pickle.loads(_action_grad)

            self.saved_action_gradients = action_grad
            action_gradient = np.array(self.saved_action_gradients)
            returns = self.calculate_discounted_returns(rewards)
            # Multiply the signal that makes actions taken more probable by the discounted
            # return of that action.  This will pull the weights in the direction that
            # makes *better* actions more probable.
            self.policy_gradient = np.zeros(action_gradient.shape)
            for t in range(0, len(returns)):
                self.policy_gradient[t] = action_gradient[t] * returns[t]

            # negate because we want gradient ascent, not descent

            self.policy.backward(-self.policy_gradient, fwd_relu1, fwd_affine1, fwd_x)
            param_key = f"{model_id}:param"
            # run an optimization step on all of the model parameters
            _params = self._model_db.get(param_key)
            if _params is None:
                params = self.policy.params
            else:
                params = pickle.loads(_params)

            self.policy.params = params
            for p in self.policy.params:
                next_w, self.policy.adam_configs[p] = adam(
                    self.policy.params[p],
                    self.policy.grads[p],
                    config=self.policy.adam_configs[p],
                )
                self.policy.params[p] = next_w
            self.policy._zero_grads()  # required every call to adam

            self._model_db.set(param_key, pickle.dumps(self.policy.params))

            # reset stuff
            del self.rewards[:]
            del self.saved_action_gradients[:]
            self._grad_db.set(grad_key, pickle.dumps(self.saved_action_gradients))

    def update_buffer(self, reward, learn=False):
        # run the environment
        if not learn:
            tmp_buffer = self.rollout(reward)
            self.policy.replay_buffer.push(tmp_buffer)
            return True
        states = []
        affine1s = []
        relu1s = []
        rewards = []
        if self.policy.replay_buffer.memory.qsize() > self.policy.batch_size:
            size = self.policy.replay_buffer.memory.qsize()
            all = self.policy.replay_buffer.memory.queue
            for j in all:
                # print(len(j))
                # if len(j)!=4:
                #    continue
                x, affine1, relu1, r = j
                states.append(x)
                affine1s.append(affine1)
                relu1s.append(relu1)
                rewards.append(r)

        return states, affine1s, relu1s, rewards

    def rollout(self, reward):

        if self.policy.temp_replay_buffer.memory.qsize() > 0:
            tmp_buffer = self.policy.temp_replay_buffer.memory.queue[0]
            tmp_buffer.append(reward)

        return tmp_buffer
