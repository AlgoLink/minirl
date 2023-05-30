import numpy as np
import gym


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, model, grad):
        if self.m is None:
            self.m = {}
            self.v = {}
            for k, v in grad.items():
                self.m[k] = np.zeros_like(v)
                self.v[k] = np.zeros_like(v)

        self.t += 1
        for k in grad.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grad[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            model[k] += self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return model, self.m, self.v


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
lr = 0.001
gamma = 0.99
eps_clip = 0.2
batch_size = 32
num_layers = 2
minibatch_size = 32
num_epochs = 10
num_iterations = 100
# Define neural network architecture
def build_policy_network():
    model = {}
    model["W1"] = np.random.randn(state_dim, 64) / np.sqrt(state_dim)
    model["b1"] = np.zeros(64)
    model["W2"] = np.random.randn(64, action_dim) / np.sqrt(64)
    model["b2"] = np.zeros(action_dim)

    def policy(obs):
        h1 = np.dot(obs, model["W1"]) + model["b1"]
        h1[h1 < 0] = 0
        logits = np.dot(h1, model["W2"]) + model["b2"]
        return np.exp(logits) / np.sum(np.exp(logits))

    def sample_action(obs):
        probs = policy(obs)
        return np.random.choice(action_dim, p=probs)

    return policy, sample_action, model


# Initialize policy network and optimizer
policy, sample_action, model = build_policy_network()
optimizer = Adam(lr=lr)

for i in range(num_iterations):
    # Collect batch of experiences
    batch_obs, batch_actions, batch_rewards, batch_dones, batch_next_obs = (
        [],
        [],
        [],
        [],
        [],
    )
    for j in range(batch_size):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = sample_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            batch_obs.append(obs)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_next_obs.append(next_obs)
            obs = next_obs
            total_reward += reward

    # Compute advantages
    values = np.zeros(batch_size)
    advantages = np.zeros(batch_size)
    for t in range(batch_size):
        obs = batch_obs[t]
        next_obs = batch_next_obs[t]
        action = batch_actions[t]
        reward = batch_rewards[t]
        done = batch_dones[t]

        value = np.sum(policy(obs) * np.arange(action_dim))
        next_value = np.sum(policy(next_obs) * np.arange(action_dim))
        values[t] = value

        if done:
            td_error = reward - value
        else:
            td_error = reward + gamma * next_value - value

        advantages[t] = td_error

    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # Update policy using PPO with Adam optimizer
    for epoch in range(num_epochs):
        indices = np.random.permutation(batch_size)
        for j in range(batch_size // minibatch_size):
            minibatch_indices = indices[j * minibatch_size : (j + 1) * minibatch_size]
            minibatch_advantages = advantages[minibatch_indices]

            minibatch_obs = np.array(batch_obs)[minibatch_indices]
            minibatch_actions = np.array(batch_actions)[minibatch_indices]
            minibatch_old_probs = policy(minibatch_obs)[
                np.arange(minibatch_size), minibatch_actions
            ]

            # Compute gradients
            grads = []
            for k in range(num_layers):
                layer_output = minibatch_obs if k == 0 else h[k - 1]
                layer_input = np.dot(layer_output, model["W"][k]) + model["b"][k]
                relu_mask = (layer_input > 0).astype(np.float32)
                layer_grad = (
                    relu_mask
                    * model["W"][k][None, :, :]
                    * minibatch_advantages[:, None, None]
                    / minibatch_old_probs[:, None, None]
                )
                grads.append(layer_grad)

            # Update parameters with Adam optimizer
            for k in range(num_layers):
                dW = np.mean(grads[k], axis=0)
                db = np.mean(grads[k], axis=(0, 1))
                grad = {"W": dW, "b": db}
                model, _ = optimizer.update(model, grad)

    # Evaluate policy
    total_reward = 0
    obs, _ = env.reset()
    done = False
    while not done:
        action = sample_action(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    print("Iteration {}: {}".format(i, total_reward))
