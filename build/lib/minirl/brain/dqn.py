import pickle
import numpy as np
import copy  # for deepcopy of model parameters


def linear(x, derivation=False):
    """
    activation function linear
    """
    if derivation:
        return 1
    else:
        return x


def relu(x, derivation=False):
    """
    activation function relu
    """
    if derivation:
        return 1.0 * (x > 0)
    else:
        return np.maximum(x, 0)


class NN:  # neural_network
    def __init__(
        self,
        input_shape,
        hidden_neurons,
        output_shape,
        learning_rate=0.01,
        model_db=None,
    ):
        self._model_db = model_db
        self.l1_weights = np.random.normal(
            scale=0.1, size=(input_shape, hidden_neurons)
        )
        self.l1_biases = np.zeros(hidden_neurons)

        self.l2_weights = np.random.normal(
            scale=0.1, size=(hidden_neurons, output_shape)
        )
        self.l2_biases = np.zeros(output_shape)

        self.learning_rate = learning_rate
        self.q_val={}

    def fit(self, x, y, epochs=1):
        """
        method implements backpropagation
        """
        for _ in range(epochs):
            # Forward propagation
            # First layer
            u1 = np.dot(x, self.l1_weights) + self.l1_biases
            l1o = relu(u1)

            # Second layer
            u2 = np.dot(l1o, self.l2_weights) + self.l2_biases
            l2o = linear(u2)

            # Backward Propagation
            # Second layer
            d_l2o = l2o - y
            d_u2 = linear(u2, derivation=True)

            g_l2 = np.dot(l1o.T, d_u2 * d_l2o)
            d_l2b = d_l2o * d_u2
            # First layer
            d_l1o = np.dot(d_l2o, self.l2_weights.T)
            d_u1 = relu(u1, derivation=True)

            g_l1 = np.dot(x.T, d_u1 * d_l1o)
            d_l1b = d_l1o * d_u1

            # Update weights and biases
            self.l1_weights -= self.learning_rate * g_l1
            self.l1_biases -= self.learning_rate * d_l1b.sum(axis=0)

            self.l2_weights -= self.learning_rate * g_l2
            self.l2_biases -= self.learning_rate * d_l2b.sum(axis=0)

        # Return actual loss
        return np.mean(np.subtract(y, l2o) ** 2)

    def predict(self, x):
        """
        method predicts q-values for state x
        """
        # First layer
        u1 = np.dot(x, self.l1_weights) + self.l1_biases
        l1o = relu(u1)

        # Second layer
        u2 = np.dot(l1o, self.l2_weights) + self.l2_biases
        l2o = linear(u2)

        return l2o

    def save_model(self, name):
        """
        method saves model
        """
        with open("{}.pkl".format(name), "wb") as model:
            pickle.dump(self, model, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        """
        method loads model
        """
        with open("{}".format(name), "rb") as model:
            tmp_model = pickle.load(model)

        self.l1_weights = tmp_model.l1_weights
        self.l1_biases = tmp_model.l1_biases

        self.l2_weights = tmp_model.l2_weights
        self.l2_biases = tmp_model.l2_biases

        self.learning_rate = tmp_model.learning_rate


    def get_model_key(self, model_id):
        return f"{model_id}:params"

    def get_weights(self, model_id):
        # return self.weights, self.biases
        l1_w, l1_b, l2_w, l2_b = self.load_weights(model_id)
        return l1_w, l1_b, l2_w, l2_b

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, l1_w, l1_b, l2_w, l2_b):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.l1_weights = copy.deepcopy(l1_w)
        self.l1_biases = copy.deepcopy(l1_b)
        self.l2_weights = copy.deepcopy(l2_w)
        self.l2_biases = copy.deepcopy(l2_b)

    def save_weights(self, model_id):
        if self._model_db is None:
            pickle.dump(
                [
                    self.l1_weights,
                    self.l1_biases,
                    self.l2_weights,
                    self.l2_biases
                ],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.get_model_key(model_id)
            self._model_db.set(
                model_key,
                pickle.dumps(
                    [
                        self.l1_weights,
                        self.l1_biases,
                        self.l2_weights,
                        self.l2_biases
                    ]
                ),
            )

    def load_weights(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            l1_w, l1_b, l2_w, l2_b = [
                self.l1_weights,
                self.l1_biases,
                self.l2_weights,
                self.l2_biases
            ]
        else:
            l1_w, l1_b, l2_w, l2_b = pickle.loads(_model)
        return l1_w, l1_b, l2_w, l2_b


class DQNAgent:
    """
    Object to handle running the algorithm. Uses a DANNetwork
    """

    def __init__(
        self, obs_dim, action_dim, hidden_layers=64, lr=0.01, gamma=0.95, model_db=None
    ) -> None:
        self._model_db = model_db
        self.gamma = gamma
        self.model = self.create_model(obs_dim, hidden_layers, action_dim, lr, model_db)

    def create_model(
        self,
        input_shape,
        hidden_neurons,
        output_shape,
        learning_rate=0.01,
        model_db=None,
    ):
        model = NN(input_shape, hidden_neurons, output_shape, learning_rate, model_db)

        return model

    def act(self, state, model_id):
        l1_w, l1_b, l2_w, l2_b = self.model.get_weights(model_id)
        self.model.set_weights(l1_w, l1_b, l2_w, l2_b)
        qvalue = self.model.predict(state)
        action = np.argmax(qvalue)

        return action

    def learn(self, state, action, next_state, reward, model_id):
        l1_w, l1_b, l2_w, l2_b = self.model.get_weights(model_id)
        self.model.set_weights(l1_w, l1_b, l2_w, l2_b)
        q_val = self.model.predict(state)
        ns_model_pred = self.model.predict(next_state)
        q_val[0][action] = reward + self.gamma * np.max(ns_model_pred[0])
        self.model.fit(state, q_val)

        self.model.q_val = q_val

        self.model.save_weights(model_id)
