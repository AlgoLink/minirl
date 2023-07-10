import numpy as np
from collections import deque
from numpy import linalg as LA
import copy  # for deepcopy of model parameters
import pickle


def relu(mat):
    return np.multiply(mat, (mat > 0))


def relu_derivative(mat):
    return (mat > 0) * 1


class NNLayer:
    # class representing a neural net layer
    def __init__(
        self, input_size, output_size, activation=None, lr=0.001, model_db=None
    ):
        self._model_db = model_db
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.activation = activation
        self._init_model(input_size, output_size, activation, lr)
        self.stored_weights = np.copy(self.weights)
        self.stored_m = np.copy(self.m)
        self.stored_v = np.copy(self.v)

    def _init_model(self, input_size, output_size, activation, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(
            low=-0.5, high=0.5, size=(input_size, output_size)
        )

        self.activation_function = activation
        self.lr = lr
        self.m = np.zeros((input_size, output_size))
        self.v = np.zeros((input_size, output_size))

        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.time = 1
        self.adam_epsilon = 0.00000001

    # Compute the forward pass for this layer
    def forward(self, inputs, remember_for_backprop=True):
        # inputs has shape batch_size x layer_input_size
        input_with_bias = np.append(inputs, 1)
        unactivated = None
        if remember_for_backprop:
            unactivated = np.dot(input_with_bias, self.weights)
        else:
            unactivated = np.dot(input_with_bias, self.stored_weights)
        # store variables for backward pass
        output = unactivated
        if self.activation_function != None:
            # assuming here the activation function is relu, this can be made more robust
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output

    def update_weights(self, gradient):
        m_temp = np.copy(self.m)
        v_temp = np.copy(self.v)

        m_temp = self.beta_1 * m_temp + (1 - self.beta_1) * gradient
        v_temp = self.beta_2 * v_temp + (1 - self.beta_2) * (gradient * gradient)
        m_vec_hat = m_temp / (1 - np.power(self.beta_1, self.time + 0.1))
        v_vec_hat = v_temp / (1 - np.power(self.beta_2, self.time + 0.1))
        self.weights = self.weights - np.divide(
            self.lr * m_vec_hat, np.sqrt(v_vec_hat) + self.adam_epsilon
        )

        self.m = np.copy(m_temp)
        self.v = np.copy(v_temp)

    def update_stored_weights(self):
        self.stored_weights = np.copy(self.weights)

    def update_time(self):
        self.time = self.time + 1

    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        # this is pointwise
        if self.activation_function != None:
            adjusted_mul = np.multiply(
                relu_derivative(self.backward_store_out), gradient_from_above
            )
        D_i = np.dot(
            np.transpose(
                np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))
            ),
            np.reshape(adjusted_mul, (1, len(adjusted_mul))),
        )
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i)
        return delta_i

    def get_model_key(self, model_id):
        return f"{model_id}:dqnws"

    def get_weights(self, model_id):
        # return self.weights, self.biases
        model_params, adam_params = self.load_weights(model_id)
        return model_params, adam_params

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, model_params, adam_params):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.weights = copy.deepcopy(model_params)
        self.m, self.v, self.time = copy.deepcopy(adam_params)

    def save_weights(self, model_id, model, m, v, time):
        if self._model_db is None:
            pickle.dump([model, [m, v, time]], open("{}.pickle".format(model_id), "wb"))
        else:
            model_key = self.get_model_key(model_id)
            self._model_db.set(model_key, pickle.dumps([model, [m, v, time]]))

    def load_weights(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            model, adam_params = self.weights, [self.m, self.v, self.time]

        else:
            model, adam_params = pickle.loads(_model)
        return model, adam_params


class RLAgent:
    # class representing a reinforcement learning agent
    env = None

    def __init__(self, config):
        self.hidden_size = config.get("hidden_size", 24)
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.num_hidden_layers = config.get("num_hidden_layers", 2)
        self.epsilon = config.get("eps", 1.0)
        self.epsilon_decay = config.get("eps_decay", 0.997)
        self.epsilon_min = config.get("eps_min", 0.01)
        self.memory = config.get("memory", deque([], 10))
        self.gamma = config.get("gamma", 0.95)
        self.lr = config.get("lr", 0.001)
        self._model_db = config.get("model_db")
        self._his_db = config.get("his_db")
        self.init_memory = self.memory

        self.layers = [
            NNLayer(
                self.input_size + 1,
                self.hidden_size,
                activation=relu,
                lr=self.lr,
                model_db=self._model_db,
            )
        ]
        for i in range(self.num_hidden_layers - 1):
            self.layers.append(
                NNLayer(
                    self.hidden_size + 1,
                    self.hidden_size,
                    activation=relu,
                    lr=self.lr,
                    model_db=self._model_db,
                )
            )
        self.layers.append(
            NNLayer(
                self.hidden_size + 1,
                self.output_size,
                lr=self.lr,
                model_db=self._model_db,
            )
        )

    def act(self, observation):
        values = self.forward(np.asmatrix(observation))
        if np.random.random() > self.epsilon:
            return np.argmax(values)
        else:
            return np.random.randint(self.env.action_space.n)

    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals

    def remember(self, done, action, observation, prev_obs):
        self.memory.append([done, action, observation, prev_obs])

    def learn(self, update_size=20, model_id=None):
        memory = self.load_his_data(model_id)
        self.set_his_data(memory)

        if len(self.memory) < update_size:
            self.memory = self.init_memory
            return
        else:
            for layer in self.layers:
                model, adam_params = layer.load_weights(model_id)
                layer.set_weights(model_params=model, adam_params=adam_params)

            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, prev_obs = self.memory[index]
                action_values = self.forward(prev_obs, remember_for_backprop=True)
                next_action_values = self.forward(new_obs, remember_for_backprop=False)
                experimental_values = np.copy(action_values)
                if done:
                    experimental_values[action_selected] = -1
                else:
                    experimental_values[action_selected] = 1 + self.gamma * np.max(
                        next_action_values
                    )
                self.backward(action_values, experimental_values)
        self.epsilon = (
            self.epsilon
            if self.epsilon < self.epsilon_min
            else self.epsilon * self.epsilon_decay
        )
        for layer in self.layers:
            layer.update_time()
            layer.update_stored_weights()
            layer.save_weights(model_id, layer.weights, layer.m, layer.v, layer.time)

    def backward(self, calculated_values, experimental_values):
        # values are batched = batch_size x output_size
        delta = calculated_values - experimental_values
        # print('delta = {}'.format(delta))
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def get_his_key(self, model_id):
        return f"{model_id}:his"

    def load_his_data(self, model_id):
        his_data_key = self.get_his_key(model_id)
        his_data = self._his_db.get(his_data_key)
        if his_data is None:
            return self.init_memory
        else:
            return pickle.loads(his_data)

    def set_his_data(self, memory):
        self.memory = copy.deepcopy(memory)

    def save_memory(self, model_id, memory):
        his_data_key = self.get_his_key(model_id)
        self._his_db.set(his_data_key, pickle.dumps(memory))


# Global variables
NUM_EPISODES = 10000
MAX_TIMESTEPS = 1000
AVERAGE_REWARD_TO_SOLVE = 195
NUM_EPS_TO_SOLVE = 100
NUM_RUNS = 20
GAMMA = 0.95
EPSILON_DECAY = 0.997
update_size = 10
hidden_layer_size = 24
num_hidden_layers = 2
model = RLAgent(env, num_hidden_layers, hidden_layer_size, GAMMA, EPSILON_DECAY)
scores_last_timesteps = deque([], NUM_EPS_TO_SOLVE)


# The main program loop
for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    if i_episode >= NUM_EPS_TO_SOLVE:
        if sum(scores_last_timesteps) / NUM_EPS_TO_SOLVE > AVERAGE_REWARD_TO_SOLVE:
            print("solved after {} episodes".format(i_episode))
            break
    # Iterating through time steps within an episode
    for t in range(MAX_TIMESTEPS):
        # env.render()
        action = model.select_action(observation)
        prev_obs = observation
        observation, reward, done, info = env.step(action)
        # Keep a store of the agent's experiences
        model.remember(done, action, observation, prev_obs)
        model.experience_replay(update_size)
        # epsilon decay
        if done:
            # If the pole has tipped over, end this episode
            scores_last_timesteps.append(t + 1)
            break
