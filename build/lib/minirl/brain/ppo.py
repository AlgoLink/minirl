import pickle
from collections import namedtuple


TrainingRecord = namedtuple("TrainingRecord", ["ep", "reward"])
Transition = namedtuple("Transition", ["s", "a", "a_log_p", "r", "s_"])

import numpy as np
import math, random
import copy  # for deepcopy of model parameters


class ActorNet2(object):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        std=5e-1,
        model_db=None,
        if_continue=False,
    ):

        print("An actor network is created.")
        self.action_cnt = output_size
        self._continue = if_continue
        self._model_db = model_db

        self.params = {}
        self.params["W1"] = self._uniform_init(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = self._uniform_init(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        self.params["W3"] = self._uniform_init(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        self.optm_cfg = {}
        self.optm_cfg["W1"] = None
        self.optm_cfg["b1"] = None
        self.optm_cfg["W2"] = None
        self.optm_cfg["b2"] = None
        self.optm_cfg["W3"] = None
        self.optm_cfg["b3"] = None

    def evaluate_gradient(self, s, a, olp, adv, b, clip_param, max_norm):

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        batch_size, _ = s.shape

        z1 = np.dot(s, W1) + b1
        H1 = np.maximum(0, z1)
        z2 = np.dot(H1, W2) + b2
        mu = b * np.tanh(z2)

        z3 = np.dot(H1, W3) + b3
        sigma = np.log(1 + np.exp(z3))

        alp = np.zeros((batch_size, len(a[0])))
        ratio = np.zeros((batch_size, len(a[0])))
        surr1 = np.zeros((batch_size, len(a[0])))
        surr2 = np.zeros((batch_size, len(a[0])))
        mu_derv = np.zeros((batch_size, len(a[0])))
        sigma_derv = np.zeros((batch_size, len(a[0])))

        for i in range(batch_size):
            for j in range(len(a[0])):
                alp[i, j] = (
                    -((a[i, j] - mu[i, j]) ** 2) / (2 * sigma[i, j] ** 2)
                    - np.log(sigma[i, j])
                    - np.log(np.sqrt(2 * np.pi))
                )
                ratio[i, j] = np.exp(alp[i, j] - olp[i, j])

                surr1[i, j] = ratio[i, j] * adv[i]
                surr2[i, j] = (
                    np.clip(ratio[i, j], 1 - clip_param, 1 + clip_param) * adv[i]
                )

                if surr2[i, j] < surr1[i, j] and (
                    ratio[i, j] < 1 - clip_param or ratio[i, j] > 1 + clip_param
                ):
                    mu_derv[i, j] = 0
                    sigma_derv[i, j] = 0
                else:
                    mu_derv[i, j] = -(
                        b
                        * adv[i]
                        * math.exp(
                            -((a[i, j] - b * math.tanh(z2[i, j])) ** 2)
                            / (2 * sigma[i, j] ** 2)
                            - olp[i, j]
                        )
                        * (1 - (math.tanh(z2[i, j])) ** 2)
                        * (a[i, j] - b * math.tanh(z2[i, j]))
                    ) / (math.sqrt(2 * math.pi) * sigma[i, j] ** 3)
                    sigma_derv[i, j] = (
                        adv[i]
                        * math.exp(
                            -((a[i, j] - mu[i, j]) ** 2)
                            / (2 * math.log(math.exp(z3[i, j]) + 1) ** 2)
                            + z3[i, j]
                            - olp[i, j]
                        )
                        * (
                            math.log(math.exp(z3[i, j]) + 1) ** 2
                            - mu[i, j] ** 2
                            + 2 * a[i, j] * mu[i, j]
                            - a[i, j] ** 2
                        )
                    ) / (
                        math.sqrt(2 * math.pi)
                        * (math.exp(z3[i, j]) + 1)
                        * math.log(math.exp(z3[i, j]) + 1) ** 4
                    )

        grads = {}

        out1 = sigma_derv.dot(W3.T)
        out1 += mu_derv.dot(W2.T)

        out1[z1 <= 0] = 0

        sigma_derv /= len(a[0])
        mu_derv /= len(a[0])
        grads["W3"] = np.dot(H1.T, sigma_derv) / batch_size
        grads["W2"] = np.dot(H1.T, mu_derv) / batch_size
        grads["W1"] = np.dot(s.T, out1) / batch_size
        grads["b3"] = np.sum(sigma_derv, axis=0) / batch_size
        grads["b2"] = np.sum(mu_derv, axis=0) / batch_size
        grads["b1"] = np.sum(out1, axis=0) / batch_size

        total_norm = np.sqrt(
            (grads["W3"] ** 2).sum()
            + (grads["W2"] ** 2).sum()
            + (grads["W1"] ** 2).sum()
            + (grads["b3"] ** 2).sum()
            + (grads["b2"] ** 2).sum()
            + (grads["b1"] ** 2).sum()
        )
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            grads["W3"] *= clip_coef
            grads["W2"] *= clip_coef
            grads["W1"] *= clip_coef
            grads["b3"] *= clip_coef
            grads["b2"] *= clip_coef
            grads["b1"] *= clip_coef
        return grads

    def train(self, s, a, olp, adv, b, clip_param, max_grad_norm):
        # Compute out and gradients using the current minibatch
        grads = self.evaluate_gradient(s, a, olp, adv, b, clip_param, max_grad_norm)
        # Update the weights using adam optimizer

        self.params["W3"] = self._adam(
            self.params["W3"], grads["W3"], config=self.optm_cfg["W3"]
        )[0]
        self.params["W2"] = self._adam(
            self.params["W2"], grads["W2"], config=self.optm_cfg["W2"]
        )[0]
        self.params["W1"] = self._adam(
            self.params["W1"], grads["W1"], config=self.optm_cfg["W1"]
        )[0]
        self.params["b3"] = self._adam(
            self.params["b3"], grads["b3"], config=self.optm_cfg["b3"]
        )[0]
        self.params["b2"] = self._adam(
            self.params["b2"], grads["b2"], config=self.optm_cfg["b2"]
        )[0]
        self.params["b1"] = self._adam(
            self.params["b1"], grads["b1"], config=self.optm_cfg["b1"]
        )[0]

        # Update the configuration parameters to be used in the next iteration
        self.optm_cfg["W3"] = self._adam(
            self.params["W3"], grads["W3"], config=self.optm_cfg["W3"]
        )[1]
        self.optm_cfg["W2"] = self._adam(
            self.params["W2"], grads["W2"], config=self.optm_cfg["W2"]
        )[1]
        self.optm_cfg["W1"] = self._adam(
            self.params["W1"], grads["W1"], config=self.optm_cfg["W1"]
        )[1]
        self.optm_cfg["b3"] = self._adam(
            self.params["b3"], grads["b3"], config=self.optm_cfg["b3"]
        )[1]
        self.optm_cfg["b2"] = self._adam(
            self.params["b2"], grads["b2"], config=self.optm_cfg["b2"]
        )[1]
        self.optm_cfg["b1"] = self._adam(
            self.params["b1"], grads["b1"], config=self.optm_cfg["b1"]
        )[1]

    def predict(self, s, b):

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        z1 = np.dot(s, W1) + b1
        H1 = np.maximum(0, z1)
        z2 = np.dot(H1, W2) + b2
        mu = b * np.tanh(z2)

        z3 = np.dot(H1, W3) + b3
        sigma = np.log(1 + np.exp(z3))

        mu = np.array([mu])
        sigma = np.array([sigma])

        # a=np.zeros((1,2))
        # alp=np.zeros((1,2))
        a = np.zeros((1, self.action_cnt))
        alp = np.zeros((1, self.action_cnt))
        # for j in range(2):
        for j in range(self.action_cnt):
            a[0, j] = mu[0, j] + sigma[0, j] * math.sqrt(
                -2.0 * math.log(random.random())
            ) * math.sin(2.0 * math.pi * random.random())
            alp[0, j] = (
                -((a[0, j] - mu[0, j]) ** 2) / (2 * sigma[0, j] ** 2)
                - np.log(sigma[0, j])
                - math.log(math.sqrt(2 * math.pi))
            )

        if not self._continue:
            probs = self._softmax(alp)
            a = np.random.choice(self.action_cnt, p=probs[0])
            # print(a,probs)

        return a, alp

    def _softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs

    def _adam(self, x, dx, config=None):
        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-4)
        config.setdefault("beta1", 0.9)
        config.setdefault("beta2", 0.999)
        config.setdefault("epsilon", 1e-8)
        config.setdefault("m", np.zeros_like(x))
        config.setdefault("v", np.zeros_like(x))
        config.setdefault("t", 0)

        next_x = None

        # Adam update formula,                                                 #
        config["t"] += 1
        config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dx
        config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * (dx**2)
        mb = config["m"] / (1 - config["beta1"] ** config["t"])
        vb = config["v"] / (1 - config["beta2"] ** config["t"])

        next_x = x - config["learning_rate"] * mb / (np.sqrt(vb) + config["epsilon"])
        return next_x, config

    def _uniform_init(self, input_size, output_size):
        u = np.sqrt(1.0 / (input_size * output_size))
        return np.random.uniform(-u, u, (input_size, output_size))

    def get_weights(self, model_id, params_type="actor"):
        # return self.weights, self.biases
        params, adam_configs = self.load_weights(model_id, params_type)
        return params, adam_configs

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, params, adam_configs):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.params = copy.deepcopy(params)
        self.optm_cfg = copy.deepcopy(adam_configs)

    def model_params_key(self, model_id, params_type="actor"):
        if params_type == "actor":
            return f"{model_id}:actorP"
        else:
            return f"{model_id}:criticP"

    def save_weights(self, model_id, params_type="actor"):
        if self._model_db is None:
            pickle.dump(
                [self.params, self.optm_cfg],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.model_params_key(model_id, params_type)
            self._model_db.set(model_key, pickle.dumps([self.params, self.optm_cfg]))

    def load_weights(self, model_id=None, params_type="actor"):
        # params_type:["actor","critic"]
        try:
            if self._model_db is None:
                params, adam_configs = pickle.load(
                    open("{}.pickle".format(model_id), "rb")
                )
            else:
                model_key = self.model_params_key(model_id, params_type)
                model = self._model_db.get(model_key)
                if model is not None:
                    params, adam_configs = pickle.loads(model)
                    return params, adam_configs
                else:
                    return self.params, self.optm_cfg

        except:
            print("Could not load weights: File Not Found, use default")
            return self.params, self.optm_cfg


class CriticNet2(object):
    def __init__(self, input_size, hidden_size, output_size, std=5e-1, model_db=None):

        print("An actor network is created.")
        self._model_db = model_db
        self.params = {}
        self.params["W1"] = self._uniform_init(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = self._uniform_init(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.optm_cfg = {}
        self.optm_cfg["W1"] = None
        self.optm_cfg["b1"] = None
        self.optm_cfg["W2"] = None
        self.optm_cfg["b2"] = None

    def evaluate_gradient(self, s, target_v, max_norm):

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        batch_size, _ = s.shape

        z1 = np.dot(s, W1) + b1
        H1 = np.maximum(0, z1)
        v = np.dot(H1, W2) + b2

        d = np.zeros((batch_size, 1))

        for i in range(batch_size):
            d[i, 0] = v[i, 0] - target_v[i]
            if d[i, 0] < -1:
                d[i, 0] = -1
            elif d[i, 0] > 1:
                d[i, 0] = 1

        grads = {}

        out1 = d.dot(W2.T)

        out1[z1 <= 0] = 0

        grads["W2"] = np.dot(H1.T, d) / batch_size
        grads["W1"] = np.dot(s.T, out1) / batch_size
        grads["b2"] = np.sum(d, axis=0) / batch_size
        grads["b1"] = np.sum(out1, axis=0) / batch_size

        total_norm = np.sqrt(
            (grads["W2"] ** 2).sum()
            + (grads["W1"] ** 2).sum()
            + (grads["b2"] ** 2).sum()
            + (grads["b1"] ** 2).sum()
        )
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            grads["W2"] *= clip_coef
            grads["W1"] *= clip_coef
            grads["b2"] *= clip_coef
            grads["b1"] *= clip_coef
        return grads

    def train(self, s, target_v, max_grad_norm):
        # Compute out and gradients using the current minibatch
        grads = self.evaluate_gradient(s, target_v, max_grad_norm)
        # Update the weights using adam optimizer

        self.params["W2"] = self._adam(
            self.params["W2"], grads["W2"], config=self.optm_cfg["W2"]
        )[0]
        self.params["W1"] = self._adam(
            self.params["W1"], grads["W1"], config=self.optm_cfg["W1"]
        )[0]
        self.params["b2"] = self._adam(
            self.params["b2"], grads["b2"], config=self.optm_cfg["b2"]
        )[0]
        self.params["b1"] = self._adam(
            self.params["b1"], grads["b1"], config=self.optm_cfg["b1"]
        )[0]

        # Update the configuration parameters to be used in the next iteration
        self.optm_cfg["W2"] = self._adam(
            self.params["W2"], grads["W2"], config=self.optm_cfg["W2"]
        )[1]
        self.optm_cfg["W1"] = self._adam(
            self.params["W1"], grads["W1"], config=self.optm_cfg["W1"]
        )[1]
        self.optm_cfg["b2"] = self._adam(
            self.params["b2"], grads["b2"], config=self.optm_cfg["b2"]
        )[1]
        self.optm_cfg["b1"] = self._adam(
            self.params["b1"], grads["b1"], config=self.optm_cfg["b1"]
        )[1]

    def predict(self, s):

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        z1 = np.dot(s, W1) + b1
        H1 = np.maximum(0, z1)
        v = np.dot(H1, W2) + b2

        return v

    def _adam(self, x, dx, config=None):
        if config is None:
            config = {}
        config.setdefault("learning_rate", 3e-4)
        config.setdefault("beta1", 0.9)
        config.setdefault("beta2", 0.999)
        config.setdefault("epsilon", 1e-8)
        config.setdefault("m", np.zeros_like(x))
        config.setdefault("v", np.zeros_like(x))
        config.setdefault("t", 0)

        next_x = None

        # Adam update formula,                                                 #
        config["t"] += 1
        config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dx
        config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * (dx**2)
        mb = config["m"] / (1 - config["beta1"] ** config["t"])
        vb = config["v"] / (1 - config["beta2"] ** config["t"])

        next_x = x - config["learning_rate"] * mb / (np.sqrt(vb) + config["epsilon"])
        return next_x, config

    def _uniform_init(self, input_size, output_size):
        u = np.sqrt(1.0 / (input_size * output_size))
        return np.random.uniform(-u, u, (input_size, output_size))

    def get_weights(self, model_id, params_type="critic"):
        # return self.weights, self.biases
        params, adam_configs = self.load_weights(model_id, params_type)
        return params, adam_configs

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, params, adam_configs):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.params = copy.deepcopy(params)
        self.optm_cfg = copy.deepcopy(adam_configs)

    def model_params_key(self, model_id, params_type="critic"):
        if params_type == "actor":
            return f"{model_id}:actorP"
        else:
            return f"{model_id}:criticP"

    def save_weights(self, model_id, params_type="critic"):
        if self._model_db is None:
            pickle.dump(
                [self.params, self.optm_cfg],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.model_params_key(model_id, params_type)
            self._model_db.set(model_key, pickle.dumps([self.params, self.optm_cfg]))

    def load_weights(self, model_id=None, params_type="critic"):
        # params_type:["actor","critic"]
        try:
            if self._model_db is None:
                params, adam_configs = pickle.load(
                    open("{}.pickle".format(model_id), "rb")
                )
            else:
                model_key = self.model_params_key(model_id, params_type)
                model = self._model_db.get(model_key)
                if model is not None:
                    params, adam_configs = pickle.loads(model)
                    return params, adam_configs
                else:
                    return self.params, self.optm_cfg

        except:
            print("Could not load weights: File Not Found, use default")
            return self.params, self.optm_cfg


class Agent:

    # clip_param = 0.2
    # max_grad_norm = 0.5

    def __init__(
        self,
        obs_dim,
        action_cnt,
        hidden_size=64,
        gamma=0.9,
        model_db=None,
        if_continue=False,
        batch_size=2,
        buffer_capacity=7,
        ppo_epoch=1,
        max_grad_norm=0.5,
        clip_param=0.2,
    ):
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.ppo_epoch = ppo_epoch
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.training_step = 0
        self._model_db = model_db
        self.manet = ActorNet2(
            obs_dim, hidden_size, action_cnt, model_db=model_db, if_continue=if_continue
        )
        self.mcnet = CriticNet2(obs_dim, hidden_size, 1, model_db=model_db)
        self.buffer = []
        self.counter = 0
        self.gamma = gamma

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def act(self, state, b=2.0):
        _action, action_log_prob = self.manet.predict(state, b)
        if not self.manet._continue:
            action = _action
        else:
            action = _action[0, 0].item()

        return action, action_log_prob

    def learn(self, buffer):
        self.training_step += 1

        # bs.shape: samples,obs_dim
        bs = np.array([t.s for t in buffer])
        d = [t.a for t in buffer]
        # ba = np.array([a for a in d])
        ba = np.array(d)
        br = np.array([t.r for t in buffer])
        bs1 = np.array([t.s_ for t in buffer])

        d = [t.a_log_p for t in buffer]
        # bolp = np.array([a for a in d])
        bolp = np.array(d)

        n = len(bs)
        k = self.batch_size

        mean = 0
        for i in range(n):
            mean += br[i]
        mean /= n
        sum = 0
        for i in range(n):
            sum += (br[i] - mean) ** 2

        std = math.sqrt(sum / (n - 1))

        target_v = np.zeros((n))
        adv = np.zeros((n))

        for i in range(n):
            br[i] = (br[i] - mean) / (std + 1e-5)

            target_v[i] = br[i] + self.gamma * self.mcnet.predict(bs1)[i, 0]
            adv[i] = target_v[i] - self.mcnet.predict(bs)[i, 0]

        for _ in range(self.ppo_epoch * 2):

            pool = np.zeros((n))
            result = np.zeros((k)).astype(int)
            for i in range(n):
                pool[i] = i
            for i in range(k):
                j = random.randint(0, n - i - 1)
                result[i] = pool[j]
                pool[j] = pool[n - i - 1]

            s = np.zeros((k, 1))
            a = np.zeros((k, 1))
            s1 = np.zeros((k, 1))
            olp = np.zeros((k, 1))
            target_v0 = np.zeros((k))
            adv0 = np.zeros((k))
            for i in range(k):
                target_v0[i] = target_v[result[i]]
                adv0[i] = adv[result[i]]

                s[i] = bs[result[i]]
                for j in range(1):
                    s[i, j] = bs[result[i]]
                # for j in range(1):
                a[i] = ba[result[i]]
                olp[i] = bolp[result[i]]

            self.manet.train(s, a, olp, adv0, 2.0, self.clip_param, self.max_grad_norm)
            self.mcnet.train(s, target_v0, self.max_grad_norm)

        del self.buffer[:]
