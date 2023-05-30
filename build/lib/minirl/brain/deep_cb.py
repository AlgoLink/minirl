import numpy as np
import pickle
import copy  # for deepcopy of model parameters


class Model:
    def __init__(self, input_dim, output_dim, hidden_dim, model_db=None):
        self._model_db = model_db
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.params = {}
        self.params["W1"] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params["b1"] = np.zeros((1, hidden_dim))
        self.params["W2"] = np.random.randn(hidden_dim, output_dim) / np.sqrt(
            hidden_dim
        )
        self.params["b2"] = np.zeros((1, output_dim))
        print("An actor network is created.")

    def forward(self, X):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        h = np.maximum(0, np.dot(X, W1) + b1)
        out = np.dot(h, W2) + b2
        return out

    def backward(self, X, y, output):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        delta_output = output - y
        delta_hidden = np.dot(delta_output, W2.T) * (W1 > 0)
        h = np.maximum(0, np.dot(X, W1) + b1)

        W2 -= 0.1 * np.dot(h.T, delta_output)
        b2 -= 0.1 * np.sum(delta_output, axis=0, keepdims=True)
        W1 -= 0.1 * np.dot(X.T, delta_hidden)
        b1 -= 0.1 * np.sum(delta_hidden, axis=0, keepdims=True)

        self.params["W1"] = W1
        self.params["b1"] = b1
        self.params["W2"] = W2
        self.params["b2"] = b2

    def get_weights(self, model_id):
        # return self.weights, self.biases
        params = self.load_weights(model_id)
        return params

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, params):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.params = copy.deepcopy(params)

    def model_params_key(self, model_id):
        return f"{model_id}:cbparams"

    def save_weights(self, model_id):
        if self._model_db is None:
            pickle.dump(
                [self.params],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.model_params_key(model_id)
            self._model_db.set(model_key, pickle.dumps(self.params))

    def load_weights(self, model_id=None):
        try:
            if self._model_db is None:
                params = pickle.load(open("{}.pickle".format(model_id), "rb"))
            else:
                model_key = self.model_params_key(model_id)
                model = self._model_db.get(model_key)
                if model is not None:
                    params = pickle.loads(model)
                    return params
                else:
                    return self.params

        except:
            print("Could not load weights: File Not Found, use default")
            return self.params


class DeepCBAgent:
    def __init__(self, obs_dim, action_cnt, hidden_layers=64, eps=0.2, model_db=None):
        self.obs_dim = obs_dim
        self.action_cnt = action_cnt
        self.eps = eps
        self._model_db = model_db

        self.model = Model(obs_dim, action_cnt, hidden_layers, model_db)

    def act(self, context, model_id):
        params = self.model.get_weights(model_id)
        self.model.set_weights(params)
        if np.random.uniform() < self.eps:
            return np.random.choice(self.model.output_dim)
        else:
            return np.argmax(self.model.forward(context))

    def learn(self, context, action, reward, model_id):
        if isinstance(context, list):
            context = np.array(context)
        y = np.zeros(self.model.output_dim)
        y[action] = reward
        output = self.model.forward(context.reshape(1, -1))

        self.model.backward(context.reshape(1, -1), y, output)
        self.model.save_weights(model_id)


class DeepCB:
    def __init__(
        self,
        feature_dim,
        action_dim,
        learning_rate=0.01,
        num_hidden_layers=1,
        hidden_layer_size=64,
    ):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size

        # initialize weight matrices for all layers
        self.weights = {}
        self.weights["W1"] = np.random.randn(feature_dim, hidden_layer_size) / np.sqrt(
            feature_dim
        )
        self.weights["b1"] = np.zeros(hidden_layer_size)
        for i in range(2, num_hidden_layers + 2):
            self.weights["W{}".format(i)] = np.random.randn(
                hidden_layer_size, hidden_layer_size
            ) / np.sqrt(hidden_layer_size)
            self.weights["b{}".format(i)] = np.zeros(hidden_layer_size)
        self.weights["WO"] = np.random.randn(hidden_layer_size, action_dim) / np.sqrt(
            hidden_layer_size
        )
        self.weights["bO"] = np.zeros(action_dim)

    def predict(self, x):
        # forward pass through network
        h = x
        for i in range(1, self.num_hidden_layers + 2):
            h = np.maximum(
                0,
                np.dot(h, self.weights["W{}".format(i)])
                + self.weights["b{}".format(i)],
            )
        output = np.dot(h, self.weights["WO"]) + self.weights["bO"]
        return output

    def update(self, x, a, r):
        # compute gradient of loss w.r.t. weights
        logits = self.predict(x)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        probs_for_actions = probs[range(len(a)), a]
        dlog = probs.copy()
        dlog[range(len(a)), a] -= 1
        dlog = dlog / len(r)
        grads = {}
        grads["WO"] = np.dot(self.weights["h"].T, dlog)
        grads["bO"] = np.sum(dlog, axis=0)
        dh = np.dot(dlog, self.weights["WO"].T)
        dh[self.weights["h"] <= 0] = 0
        for i in reversed(range(1, self.num_hidden_layers + 1)):
            grads["W{}".format(i)] = np.dot(self.weights["H{}".format(i - 1)].T, dh)
            grads["b{}".format(i)] = np.sum(dh, axis=0)
            dh = np.dot(dh, self.weights["W{}".format(i)].T)
            dh[self.weights["H{}".format(i - 1)] <= 0] = 0
        grads["W1"] = np.dot(x.T, dh)
        grads["b1"] = np.sum(dh, axis=0)

        # update weights using Adam optimizer
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for param in self.weights:
            m = np.zeros_like(self.weights[param])
            v = np.zeros_like(self.weights[param])
            t = 0
            for i in range(grads[param].shape[0]):
                g = grads[param][i, :]
                t += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g**2)
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                self.weights[param][i, :] -= (
                    self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                )

    def select_action(self, x):
        # choose action with highest expected reward based on current context
        logits = self.predict(x)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.argmax(probs)
