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
# TODO: add weight saving and loading?
import numpy as np
import pickle
import copy  # for deepcopy of model parameters

from ..common.optim import adam


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
        hidden_dim=64,
        lr=1e-3,
        model_db=None,
        his_db=None,
        score_db=None,
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
        self.lr = lr

        self._model_db = model_db
        self._his_db = his_db
        self._score_db = score_db

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
    def _zero_grads2(self):
        """Reset gradients to 0. This should be called during optimization steps"""
        for g in self.grads:
            self.grads[g] = np.zeros_like(self.grads[g])

    def grads_key(self, model_id):
        return f"{model_id}:grads"

    def save_grads(self, grads, model_id):
        self._score_db.set(self.grads_key(model_id), pickle.dumps(grads))

    def get_grads(self, model_id):
        _grads = self._score_db.get(self.grads_key(model_id))
        if _grads is None:
            grads = self.grads
        else:
            grads = pickle.loads(_grads)
        return grads

    def _zero_grads(self, model_id=None):
        """Reset gradients to 0. This should be called during optimization steps"""
        if self._score_db is None:
            self._zero_grads2()
        else:
            grads = self.get_grads(model_id)
            for g in grads:
                grads[g] = np.zeros_like(grads[g])

            self.save_grads(grads, model_id)
            self.grads = grads

    def _add_to_cache2(self, name, val):
        """Helper function to add a parameter to the cache without having to do checks"""
        if name in self.cache:
            self.cache[name].append(val)
        else:
            self.cache[name] = [val]

    def cache_key(self, model_id):
        return f"{model_id}:cache"

    def cache_local_key(self, name, model_id):
        global_key = self.cache_key(model_id)
        return f"{global_key}:{name}"

    def get_cache(self, model_id):
        _cache = self._score_db.get(self.cache_key(model_id))
        if _cache is None:
            cache = {}
        else:
            cache = pickle.loads(_cache)

        return cache

    def save_cache(self, cache, model_id):
        self._score_db.set(self.cache_key(model_id), pickle.dumps(cache))

    def _add_to_cache(self, name, val, model_id):
        """Helper function to add a parameter to the cache without having to do checks"""
        if self._score_db is None:
            self._add_to_cache2(name, val)
            return
        cache = self.get_cache(model_id)
        if name in cache:
            cache[name].append(val)
        else:
            cache[name] = [val]

        self.save_cache(cache, model_id)

    def _add_to_cache_once(self, cache, name, val):
        """Helper function to add a parameter to the cache without having to do checks"""

        if name in cache:
            cache[name].append(val)
        else:
            cache[name] = [val]

        return cache

    def _add_to_cache_using_rpush(self, name, val, model_id):
        """key in ['fwd_relu1', 'fwd_affine1', 'fwd_x']"""
        global_key = self.cache_key(model_id)
        local_key = f"{global_key}:{name}"
        if name == "rewards":
            self._score_db.rpush(local_key, str(val))
        else:
            self._score_db.rpush(local_key, pickle.dumps(val))

    def _get_cache_using_lrange(self, name, model_id, get_n="-1"):
        global_key = self.cache_key(model_id)
        local_key = f"{global_key}:{name}"
        return self._score_db.lrange(local_key, "0", get_n)

    def _update_grad(self, name, val, model_id=None):
        """Helper fucntion to set gradient without having to do checks"""
        if self._score_db is None:
            self._update_grad2(name, val)
            return

        grads = self.get_grads(model_id)
        if name in grads:
            grads[name] += val
        else:
            grads[name] = val

        self.save_grads(grads, model_id)
        # self.grads=

    def _update_grad2(self, name, val):
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
        # self._add_to_cache("fwd_x", x)
        # self._add_to_cache("fwd_affine1", affine1)
        # self._add_to_cache("fwd_relu1", relu1)
        return probs, affine1, relu1

    def backward(self, dout, model_id=None):
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
        cache = self.get_cache(model_id)
        _fwd_relu1 = cache["fwd_relu1"]
        _fwd_affine1 = cache["fwd_affine1"]
        _fwd_x = cache["fwd_x"]
        fwd_relu1 = np.concatenate(_fwd_relu1)
        fwd_affine1 = np.concatenate(_fwd_affine1)
        fwd_x = np.concatenate(_fwd_x)

        # fwd_relu1 = np.concatenate(self.cache["fwd_relu1"])
        # wd_affine1 = np.concatenate(self.cache["fwd_affine1"])
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
        # self._update_grad("W1", dW1)
        # self._update_grad("b1", db1)
        # self._update_grad("W2", dW2)
        # self._update_grad("b2", db2)
        self._update_grad2("W1", dW1)
        self._update_grad2("b1", db1)
        self._update_grad2("W2", dW2)
        self._update_grad2("b2", db2)

        # reset cache for next backward pass
        self.cache = {}
        # self.save_cache({}, model_id)

    def get_weights(self, model_id):
        # return self.weights, self.biases
        params, adam_configs = self.load_weights(model_id)
        return params, adam_configs

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, params, adam_configs):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.params = copy.deepcopy(params)
        self.adam_configs = copy.deepcopy(adam_configs)

    def model_params_key(self, model_id):
        return f"{model_id}:params"

    def save_weights(self, model_id):
        if self._model_db is None:
            pickle.dump(
                [self.params, self.adam_configs],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.model_params_key(model_id)
            self._model_db.set(
                model_key, pickle.dumps([self.params, self.adam_configs])
            )

    def load_weights(self, model_id=None):
        try:
            if self._model_db is None:
                params, adam_configs = pickle.load(
                    open("{}.pickle".format(model_id), "rb")
                )
            else:
                model_key = self.model_params_key(model_id)
                model = self._model_db.get(model_key)
                if model is not None:
                    params, adam_configs = pickle.loads(model)
                    return params, adam_configs
                else:
                    return self.params, self.adam_configs

        except:
            print("Could not load weights: File Not Found, use default")
            return self.params, self.adam_configs


class PGAgent(object):
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
    """

    def __init__(
        self,
        obs_n,
        act_n,
        hidden_dim=64,
        gamma=0.99,
        lr=1e-3,
        seed=42,
        model_db=None,
        score_db=None,
        his_db=None,
        model_id=None,
    ):
        self.policy = self.create_model(
            obs_n, act_n, hidden_dim, lr, model_db, score_db, his_db
        )
        self.target_model = self.create_model(
            obs_n, act_n, hidden_dim, lr, model_db, score_db, his_db
        )
        self.target_model.set_weights(*self.policy.get_weights(model_id))

        # RL specific bookkeeping
        self.saved_action_gradients = []
        self.rewards = []
        self.gamma = gamma
        np.random.seed(seed)

    def create_model(
        self, obs_dim, num_actions, hidden_layers, lr, model_db, score_db, his_db
    ):
        model = PolicyNetwork(
            ob_n=obs_dim,
            ac_n=num_actions,
            hidden_dim=hidden_layers,
            lr=lr,
            model_db=model_db,
            his_db=his_db,
            score_db=score_db,
        )

        return model

    def act(self, obs, model_id, target=True, save_aprobs=True, return_details=False):
        """
        Pass observations through network and sample an action to take. Keep track
        of dh to use to update weights
        """
        self.target_model.set_weights(*self.policy.get_weights(model_id))
        if target:
            model = self.target_model
        else:
            model = self.policy
        obs = np.reshape(obs, [1, -1])
        netout, affine1, relu1 = model.forward(obs)  # [0]

        probs = netout[0]
        # randomly sample action based on probabilities
        action = np.random.choice(self.policy.ac_n, p=probs)
        # derivative that pulls in direction to make actions taken more probable
        # this will be fed backwards later
        # (see README.md for derivation)
        dh = -1 * probs
        dh[action] += 1
        # self.saved_action_gradients.append(dh)
        if save_aprobs:
            model._add_to_cache_using_rpush("aprobs", dh, model_id)

            cache = model.get_cache(model_id)
            cache = model._add_to_cache_once(cache, "fwd_x", obs)
            cache = model._add_to_cache_once(cache, "fwd_affine1", affine1)
            cache = model._add_to_cache_once(cache, "fwd_relu1", relu1)
            model.save_cache(cache, model_id)

        if return_details:
            return action, dh, obs, affine1, relu1, model

        return action

    def save_act_details(self, dh, obs, affine1, relu1, model, model_id):
        model._add_to_cache_using_rpush("aprobs", dh, model_id)
        cache = model.get_cache(model_id)
        cache = model._add_to_cache_once(cache, "fwd_x", obs)
        cache = model._add_to_cache_once(cache, "fwd_affine1", affine1)
        cache = model._add_to_cache_once(cache, "fwd_relu1", relu1)
        model.save_cache(cache, model_id)

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
        # normalize for better statistical properties /baseline
        returns = (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps
        )
        return returns

    def empty_cache(self, model_id):
        reward_local_key = self.target_model.cache_local_key("rewards", model_id)
        aprobs_local_key = self.target_model.cache_local_key("aprobs", model_id)
        self.target_model._score_db.delete(reward_local_key)
        self.target_model._score_db.delete(aprobs_local_key)
        cache_key = self.target_model.cache_key(model_id)
        self.target_model._score_db.delete(cache_key)

    def learn(self, model_id, target=True):
        """
        At the end of the episode, calculate the discounted return for each time step and update the model parameters
        """
        self.target_model.set_weights(*self.policy.get_weights(model_id))
        if target:
            model = self.target_model
        else:
            model = self.policy

        # upload grads
        # model.grads = model.get_grads(model_id)
        # action_gradient = np.array(self.saved_action_gradients)
        _aprobs = model._get_cache_using_lrange("aprobs", model_id)
        aprobs = [pickle.loads(aprob) for aprob in _aprobs]
        action_gradient = np.array(aprobs)

        _rewards = model._get_cache_using_lrange("rewards", model_id)
        rewards = [float(r) for r in _rewards]
        # returns = self.calculate_discounted_returns(self.rewards)
        returns = self.calculate_discounted_returns(rewards)
        # Multiply the signal that makes actions taken more probable by the discounted
        # return of that action.  This will pull the weights in the direction that
        # makes *better* actions more probable.
        self.policy_gradient = np.zeros(action_gradient.shape)
        for t in range(0, len(returns)):
            self.policy_gradient[t] = action_gradient[t] * returns[t]

        # negate because we want gradient ascent, not descent
        model.backward(-self.policy_gradient, model_id)

        del self.policy_gradient

        # run an optimization step on all of the model parameters
        for p in model.params:
            next_w, model.adam_configs[p] = adam(
                model.params[p],
                model.grads[p],
                config=model.adam_configs[p],
            )
            model.params[p] = next_w
        model._zero_grads2()  # required every call to adam

        # reset stuff
        del self.rewards[:]
        del self.saved_action_gradients[:]
        self.empty_cache(model_id)

        model.save_weights(model_id)
