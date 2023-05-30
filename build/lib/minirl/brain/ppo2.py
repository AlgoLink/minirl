import numpy as np


class Net:
    def __init__(self, input_size, output_size, hidden_layer_size=64):
        # initialize weights single net
        self.params = {}
        self.params["W1"] = self._uniform_init(input_size, hidden_layer_size)
        self.params["b1"] = np.zeros(hidden_layer_size)

        self.params["W2"] = self._uniform_init(hidden_layer_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def _uniform_init(self, input_size, output_size):
        u = np.sqrt(1.0 / (input_size * output_size))
        return np.random.uniform(-u, u, (input_size, output_size))

    def policy(self, observation):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        h = np.matmul(observation, W1) + b1
        h[h < 0] = 0  # ReLU activation
        logits = np.matmul(h, W2) + b2
        # prob = np.exp(logits) / np.sum(np.exp(logits))  # softmax activation
        prob = self._softmax(logits)
        return prob

    def _softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs

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

    # apply gradients to model weights using Adam
    def apply_gradients(
        self, model, grads, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        for param in model:
            if param.startswith("W"):
                m = np.zeros_like(model[param])
                v = np.zeros_like(model[param])
                t = 0
                for i in range(grads[param].shape[0]):
                    g = grads[param][i, :]
                    t += 1
                    m = beta1 * m + (1 - beta1) * g
                    v = beta2 * v + (1 - beta2) * (g**2)
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    model[param][i, :] -= (
                        learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    )
            else:
                m = np.zeros_like(model[param])
                v = np.zeros_like(model[param])
                t = 0
                for i in range(grads[param].shape[0]):
                    g = grads[param][i]
                    t += 1
                    m = beta1 * m + (1 - beta1) * g
                    v = beta2 * v + (1 - beta2) * (g**2)
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    model[param][i] -= (
                        learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    )
        return model


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        action_cnt,
        hidden_layer=64,
        lr=0.001,
        gamma=0.99,
        lam=0.95,
        epsilon=0.2,
        epochs=1,
        batch_size=1,
    ):
        self.lr = lr  # learning rate
        self.gamma = gamma  # discount factor
        self.lam = 0.95  # Generalized Advantage Estimation
        self.eps = epsilon  # PPO clipping parameter

        self.batch_size = batch_size
        self.epochs = epochs

        self.actor = Net(obs_dim, action_cnt, hidden_layer)
        self.critic = Net(obs_dim, 1, hidden_layer)

    def act(self, state):
        probs = self.actor.policy(state)
        value = self.critic.predict(state)

        return probs, value

    def learn(self, buffer):

        for epoch in range(self.epochs):
            # shuffle data
            observations = buffer["states"]
            actions = buffer["actions"]
            advantages = buffer["advantages"]
            returns = buffer["returns"]
            values = buffer["values"]
            indices = np.arange(len(observations))
            np.random.shuffle(indices)

            for i in range(len(observations) // self.batch_size):
                # mini-batch
                idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
                batch_observations = observations[idx]
                batch_actions = actions[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # compute loss and gradients
                action_probs = self.actor.policy(batch_observations)
                new_log_probs = np.log(np.sum(action_probs * batch_actions, axis=1))
                old_log_probs = np.log(
                    np.sum(
                        self.actor.policy(batch_observations) * batch_actions, axis=1
                    )
                )
                ratio = np.exp(new_log_probs - old_log_probs)
                clipped_ratio = np.clip(ratio, 1 - self.eps, 1 + self.eps)

                surrogate_1 = ratio * batch_advantages
                surrogate_2 = clipped_ratio * batch_advantages
                policy_loss = -np.mean(np.minimum(surrogate_1, surrogate_2))
                value_loss = np.mean((batch_returns - values) ** 2)
                entropy_loss = -np.mean(
                    np.sum(action_probs * np.log(action_probs), axis=1)
                )

                # compute gradients using backpropagation
                grad_logits = (
                    action_probs - batch_actions
                )  # gradient of the log softmax output
                grad_b2 = np.sum(grad_logits, axis=0)  # bias gradient
                grad_W2 = np.matmul(hidden.T, grad_logits)  # weight gradient
                grad_hidden = np.matmul(grad_logits, W2.T)  # gradient wrt hidden state
                grad_hidden[hidden <= 0] = 0  # ReLU gradient
                grad_b1 = np.sum(grad_hidden, axis=0)  # bias gradient
                grad_W1 = np.matmul(
                    batch_observations.T, grad_hidden
                )  # weight gradient

                # apply gradients to update weights
                W1 += lr * grad_W1
                b1 += lr * grad_b1
                W2 += lr * grad_W2
                b2 += lr * grad_b2
