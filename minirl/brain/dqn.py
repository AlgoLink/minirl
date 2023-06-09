import numpy as np
import copy  # for deepcopy of model parameters
import pickle
import random
from collections import deque

from .dqn_base import Policy


class FCNPolicy(Policy):
    # For weight initialization, used He normal init.
    # For bias initialization, used Zero init.
    def __init__(self, input_n, output_n, hidden_n=16, hidden_layer_n=1, model_db=None):
        self._model_db = model_db
        self.layers = list()
        input_w = np.random.normal(scale=np.sqrt(2 / input_n), size=(input_n, hidden_n))
        input_b = np.zeros(hidden_n)

        self.layers.append((input_w, input_b))

        for i in range(hidden_layer_n):
            hidden_w = np.random.normal(
                scale=np.sqrt(2 / hidden_n), size=(hidden_n, hidden_n)
            )
            hidden_b = np.zeros(hidden_n)

            self.layers.append((hidden_w, hidden_b))

        output_w = np.random.normal(
            scale=np.sqrt(2 / hidden_n), size=(hidden_n, output_n)
        )
        output_b = np.zeros(output_n)

        self.layers.append((output_w, output_b))

    def train(self, x, y, learning_rate=0.00003):
        predict, update_helper = self.predict(x, update_mode=True)
        update_layers = list()
        d = predict - y
        cost = np.mean(np.square(d))
        d = d * 2
        reversed_layers = list(reversed(self.layers))

        for i, info in enumerate(update_helper):
            # C is cost, x is node
            prev_layer, mid_layer, layer, activate_d_func = info
            # dC/df'(wx+b)
            dl = d
            if activate_d_func is None:
                dl_a = dl
            else:
                # Calculate f'(wx + b) * dC/df'(wx+b)
                dl_a = activate_d_func(mid_layer) * dl

            # dC/db = 1 * f'(wx + b) * dC/df'(wx+b)
            db = np.mean(dl_a, axis=0)
            # dC/dw = x * f'(wx + b) * dC/df'(wx+b)
            # y.shape[0] for mean of total gradient
            dw = prev_layer.T @ dl_a / y.shape[0]

            w, b = reversed_layers[i]

            # For backpropagation, we need tensor shape of (batchsize, output) which has each node's gradient for each batch.
            # dC/dx = df(wx+b)/dx * dC/df(wx+b) = w * f'(wx + b) * dC/df'(wx+b) = w * dl_a
            d = (w @ dl_a.T).T
            update_layers.append((dw, db))

        update_layers.reverse()

        for i, l in enumerate(zip(update_layers, self.layers)):
            update_layer, layer = l
            dw, db = update_layer
            w, b = layer
            # use for clipping gradient
            # dw = np.clip(dw, -0.5, 0.5)
            # db = np.clip(db, -0.5, 0.5)
            w -= learning_rate * dw
            b -= learning_rate * db
            self.layers[i] = (w, b)

        return cost

    def predict(self, x, update_mode=False):
        update_helper = list()
        prev_x = x
        for param in self.layers[:-1]:
            w, b = param
            mid_x = x @ w + b
            x = self.ReLU(mid_x)
            if update_mode:
                update_helper.append((prev_x, mid_x, x, self.d_ReLU))
                prev_x = np.copy(x)

        w, b = self.layers[-1]
        if update_mode:
            mid_x = x @ w + b
            update_helper.append((x, mid_x, mid_x, None))
            return x @ w + b, list(reversed(update_helper))
        else:
            return x @ w + b

    def get_model_key(self, model_id):
        return f"{model_id}:dqnwgts"

    def get_weights(self, model_id):
        # return self.weights, self.biases
        layers = self.load_weights(model_id)
        return layers

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, layers):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.layers = copy.deepcopy(layers)

    def save_weights(self, model_id):
        if self._model_db is None:
            pickle.dump(self.layers, open("{}.pickle".format(model_id), "wb"))
        else:
            model_key = self.get_model_key(model_id)
            self._model_db.set(model_key, pickle.dumps(self.layers))

    def load_weights(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            layers = self.layers

        else:
            layers = pickle.loads(_model)
        return layers


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_n=16,
        hidden_layer_n=1,
        eps=0.2,
        gamma=0.99,
        batch_size=1,
        epoch=1,
        replay_memory_len=100,
        model_db=None,
    ) -> None:
        self.eps = eps
        self.gamma = gamma
        self.action_cnt = action_dim
        self.state_shape = obs_dim
        self.batch_size = batch_size
        self.epoch = epoch
        self.replay_memory_len = replay_memory_len
        self.memory = deque()

        self.model = FCNPolicy(
            input_n=obs_dim,
            output_n=action_dim,
            hidden_n=hidden_n,
            hidden_layer_n=hidden_layer_n,
            model_db=model_db,
        )

    def act(self, state, model_id):
        # e-greedy(Q)
        if np.random.randn() < self.eps:
            action = np.random.randint(self.action_cnt)
        else:
            layers = self.model.load_weights(model_id)
            self.model.set_weights(layers)

            q_vals = self.model.predict(state)
            action = np.argmax(q_vals)

        return action

    def learn(self, state, action, next_state, reward, model_id, done=False):
        batch_size = self.batch_size
        epoch = self.epoch
        state_shape = self.state_shape
        action_size = self.action_cnt
        if len(self.memory) < self.replay_memory_len:
            self.add_sample(state, next_state, action, reward, done)
            return {"msg": "not enough,model is not learned"}
        # X = np.zeros((batch_size, state_shape))
        # y = np.zeros((batch_size, action_size))
        layers = self.model.load_weights(model_id)
        self.model.set_weights(layers)
        for i in range(1, epoch + 1):
            states, next_states, action, reward, terminal = self._sample_memory()

            next_states = np.array(next_states)
            target_Q_value = self.model.predict(next_states)
            states = np.array(states)
            Q_value = self.model.predict(states)
            Y = list()
            for j in range(self.batch_size):
                if terminal[j]:
                    Q_value[j, action[j]] = reward[j]
                    Y.append(Q_value[j])
                else:
                    Q_value[j, action[j]] = reward[j] + self.gamma * np.max(
                        target_Q_value[j]
                    )
                    Y.append(Q_value[j])

            Y = np.array(Y)
            cost = self.model.train(states, Y)

        self.model.save_weights(model_id)
        print(cost, "cost", len(self.memory))
        self.memory = []
        print(cost, "cost", len(self.memory))
        return cost

    def add_sample(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))
        if len(self.memory) > self.replay_memory_len:
            self.memory.popleft()

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.batch_size)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal
