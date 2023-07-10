import numpy as np
import warnings

warnings.filterwarnings("error")  # for overflow


# differentiable parameterized policy
class Policy:
    parameters = None

    def __init__(self, alpha, obs_dim, actions_n, clip=None):
        self.alpha = alpha
        self.parameters = np.zeros(
            (actions_n, obs_dim)
        )  # So I don't have to keep transposing later
        self.clip = clip

    def get_action(self, state_features, compute_gradient=False):
        state_features = np.squeeze(state_features)
        # softmax over all actions - probability of taking them
        action_values = np.exp(np.matmul(self.parameters, state_features))
        normalization = np.sum(action_values, axis=0)
        action_probabilities = action_values / normalization

        # sample an action
        action_index = np.random.choice(
            np.arange(action_probabilities.shape[0]), 1, p=action_probabilities
        ).item()

        if compute_gradient:
            # compute gradient based on rules derived in class
            state_features = np.reshape(state_features, (1, -1))
            action_probabilities = np.reshape(action_probabilities, (-1, 1))

            ln_gradient = -1 * action_probabilities * state_features
            ln_gradient[action_index] = (
                1 - action_probabilities[action_index]
            ) * state_features
            self.ln_gradient = ln_gradient

        return (action_index, np.squeeze(action_probabilities))

    def compute_gd(self, state, action_index, action_probabilities):
        # compute gradient based on rules derived in class
        state_features = np.reshape(state, (1, -1))
        action_probabilities = np.reshape(action_probabilities, (-1, 1))

        ln_gradient = -1 * action_probabilities * state_features
        ln_gradient[action_index] = (
            1 - action_probabilities[action_index]
        ) * state_features

        return ln_gradient

    def ppo_gradient_step(
        self, state_features, old_action_probabilities, actions, advantages
    ):
        n_len = state_features.shape[0]
        s_len = state_features.shape[1]
        a_len = old_action_probabilities.shape[1]

        # get probability for every action at every state that was seen in the previous rollout
        state_features = np.expand_dims(np.transpose(state_features), 0)
        parameters = np.reshape(self.parameters, (a_len, s_len, 1))
        action_values = np.exp(np.sum(parameters * state_features, axis=1))
        normalization = np.sum(action_values, axis=0)
        current_action_probabilities = np.reshape(
            action_values / normalization, (a_len, 1, n_len)
        )

        # compute ratio needed by the ppo algorithm
        old_action_probabilities = np.expand_dims(
            np.transpose(old_action_probabilities), 1
        )
        ratios = current_action_probabilities / old_action_probabilities

        gradients = (
            -1 * current_action_probabilities
        ) * state_features  # start with "not action" gradient first

        # shameful loop solution
        # both corrects the gradient for which actions were actually taken and zeros out gradients that violate our clip
        actions = np.squeeze(actions)
        for i in range(n_len):
            action = actions[i]
            gradients[action, :, i] = (
                1 - current_action_probabilities[action, 0, i]
            ) * state_features[0, :, i]
            for a in range(a_len):
                if not (
                    (advantages[i] >= 0 and ratios[a, 0, i] < 1 + self.clip)
                    or (advantages[i] < 0 and ratios[a, 0, i] > 1 - self.clip)
                ):
                    gradients[a, :, i] = np.zeros(s_len)

        # compute final gradient. Notice that the zero gradients stay as zero no matter the advantage
        advantages = np.reshape(advantages, (1, 1, n_len))
        gradients = gradients * advantages * ratios
        gradients = np.mean(gradients, axis=2)

        self.parameters += self.alpha * gradients

    def update_parameters(self, td_error, ln_gradient):
        self.parameters += self.alpha * td_error * ln_gradient


# differentiable value function approximation
class ValueFunction:
    parameters = None

    def __init__(self, alpha, obs_dim):
        self.alpha = alpha
        self.parameters = np.zeros(obs_dim)

    def evaluate_state(self, state_features):
        return np.squeeze(
            np.matmul(state_features, np.reshape(self.parameters, (-1, 1)))
        )

    def update_parameters(self, td_error, state_features):
        self.parameters += self.alpha * np.mean(
            np.reshape(td_error, (-1, 1)) * state_features, axis=0
        )


class OnePPOAgent:
    def __init__(
        self,
        obs_dim,
        actions_n,
        policy_alpha=0.1,
        critic_alpha=0.01,
        discount=0.97,
        clip=None,
    ):
        self.policy_alpha = policy_alpha
        self.critic_alpha = critic_alpha
        self.clip = clip
        self.discount = discount
        self.obs_dim = obs_dim
        self.actions_n = actions_n

        self._init_model()

    def _init_model(self):
        self.policy = Policy(
            alpha=self.policy_alpha,
            obs_dim=self.obs_dim,
            actions_n=self.actions_n,
            clip=self.clip,
        )

        self.critic = ValueFunction(alpha=self.critic_alpha, obs_dim=self.obs_dim)

    def act(self, state, compute_gradient=False):
        # draw action from policy
        action, action_probabilities = self.policy.get_action(state, compute_gradient)

        return action, action_probabilities

    def learn_one_ac(self, state, action_probs, action, reward, next_state):
        # One Step actor critic
        state_value = self.critic.evaluate_state(state)
        # compute TD error
        next_state_value = self.critic.evaluate_state(next_state)
        target = reward + self.discount * next_state_value
        td_error = target - state_value

        # update actor & critic
        ln_gradient = self.policy.compute_gd(
            state=state, action_index=action, action_probabilities=action_probs
        )
        self.policy.update_parameters(td_error, ln_gradient)
        self.critic.update_parameters(td_error, state)

    def learn_one_ppo(self, state, action_probs, actions, next_state, rewards):
        # ppo works by running the environment for some steps and then computing stochastic gradient descent on the collected s,a,r,s' examples
        # I use a value function for the baseline just like in actor-critic
        state_features = np.array(state)
        next_state_features = np.array(next_state)
        action_probabilities = np.array(action_probs)
        actions = np.array(actions)
        rewards = np.array(rewards)

        state_values = np.squeeze(self.critic.evaluate_state(state_features))
        next_state_values = np.squeeze(self.critic.evaluate_state(next_state_features))
        targets = (
            rewards + self.discount * next_state_values
        )  # super easy td error calculation
        advantages = targets - state_values

        self.policy.ppo_gradient_step(
            state_features, action_probabilities, actions, advantages
        )

        errors = targets - state_values
        self.critic.update_parameters(errors, state_features)


def one_step_actor_critic(
    mdp,
    value_function_alpha,
    policy_alpha,
    iterations,
    episodes,
    max_actions,
    final_performance_episodes,
):
    # print("One Step actor critic with mdp: " + mdp.name)

    actions_vs_episodes_all = np.zeros((iterations, episodes))
    for i in range(iterations):

        policy = Policy(policy_alpha, mdp.state_features_length, mdp.actions_length)
        value_function = ValueFunction(value_function_alpha, mdp.state_features_length)

        # print("Iteration: " + str(i))
        actions_vs_episodes = []
        for episode in range(episodes):

            state = mdp.get_start_state()
            state_features = mdp.get_state_features(state)
            # print(state_features)
            state_value = value_function.evaluate_state(state_features)
            actions = 0

            while not mdp.episode_over(state):
                actions += 1
                if actions == max_actions:
                    break
                # draw action from policy
                action, action_probabilities = policy.get_action(state_features, True)

                # execute action a, observe r and s'
                next_state = mdp.get_next_state(state, action)
                reward = mdp.get_reward(next_state)

                # get state features from state
                next_state_features = mdp.get_state_features(next_state)

                # compute TD error
                next_state_value = value_function.evaluate_state(next_state_features)
                target = reward + mdp.discount * next_state_value
                td_error = target - state_value

                # update actor & critic
                policy.update_parameters(td_error, state_features)
                value_function.update_parameters(td_error, state_features)

                # s = s'
                state = next_state
                state_features = next_state_features
                state_value = next_state_value

            actions_vs_episodes.append(actions)
        actions_vs_episodes_all[i] = np.array(actions_vs_episodes)

    return collect_statistics(actions_vs_episodes_all, final_performance_episodes)


# ppo works by running the environment for some steps and then computing stochastic gradient descent on the collected s,a,r,s' examples
# I use a value function for the baseline just like in actor-critic
def proximal_policy_optimization(
    mdp,
    value_function_alpha,
    policy_alpha,
    clip,
    iterations,
    episodes,
    max_actions,
    rollout_episodes,
    epochs,
    final_performance_episodes,
):
    # print("Proximal Policy Optimization with mdp: " + mdp.name)
    actions_vs_episodes_all = np.zeros((iterations, episodes))

    for i in range(iterations):
        # print("Iteration: " + str(i))
        policy = Policy(
            policy_alpha, mdp.state_features_length, mdp.actions_length, clip
        )
        value_function = ValueFunction(value_function_alpha, mdp.state_features_length)
        actions_vs_episodes = []

        episode = 0
        while episode < episodes:
            current_rollout_episodes = min(episodes - episode, rollout_episodes)
            # run the environment
            (
                state_features,
                actions,
                action_probabilities,
                targets,
                advantages,
                episode_lengths,
            ) = ppo_rollout(
                mdp, policy, value_function, current_rollout_episodes, max_actions
            )

            episode += rollout_episodes
            actions_vs_episodes += episode_lengths

            # SGD on policy and value function
            for j in range(epochs):
                policy.ppo_gradient_step(
                    state_features, action_probabilities, actions, advantages
                )

                state_values = value_function.evaluate_state(state_features)
                errors = targets - state_values
                value_function.update_parameters(errors, state_features)

        # going to take average over all ppo iterations
        actions_vs_episodes_all[i] = np.array(actions_vs_episodes)

    return collect_statistics(actions_vs_episodes_all, final_performance_episodes)


def ppo_rollout(mdp, policy, value_function, rollout_episodes, max_actions):

    state_features = []
    actions = []
    action_probabilities = []
    rewards = []

    last_state_features = []
    episode_lengths = []

    # collect rollout_episodes trajectories
    for ep in range(rollout_episodes):
        state = mdp.get_start_state()
        state_feature = mdp.get_state_features(state)
        actions_taken = 0

        while not mdp.episode_over(state):
            if actions_taken == max_actions:
                break
            # draw action from policy
            action, action_probability = policy.get_action(state_feature, False)

            # execute action a, observe r and s'
            next_state = mdp.get_next_state(state, action)
            reward = mdp.get_reward(next_state)

            # record for stachastic gradient descent later
            state_features.append(state_feature[0])
            actions.append(action)
            action_probabilities.append(action_probability)
            rewards.append(reward)

            # s = s'
            state = next_state
            state_feature = mdp.get_state_features(state)

            actions_taken += 1

        last_state_features.append(
            state_feature
        )  # v(s') for the last example in this episode batch (need for Advantage function)
        episode_lengths.append(actions_taken)

    state_features = np.array(state_features)
    actions = np.array(actions)
    action_probabilities = np.array(action_probabilities)
    rewards = np.array(rewards)

    # make sure state_values and next_state_values line up
    ep_begin = 0
    state_values = np.squeeze(value_function.evaluate_state(state_features))
    last_state_values = np.reshape(
        np.squeeze(value_function.evaluate_state(last_state_features)), (-1)
    )
    next_state_values = np.zeros(state_values.shape)
    # this makes sure the state_value array and next_next_value arrays are lined up so that the td errors can easily be computed
    for i in range(rollout_episodes):
        l = episode_lengths[i]
        next_state_values[ep_begin : ep_begin + l - 1] = state_values[
            ep_begin + 1 : ep_begin + l
        ]
        next_state_values[ep_begin + l - 1] = last_state_values[i]
        ep_begin += l

    targets = (
        rewards + mdp.discount * next_state_values
    )  # super easy td error calculation
    advantages = targets - state_values

    return (
        state_features,
        actions,
        action_probabilities,
        targets,
        advantages,
        episode_lengths,
    )


# basic statistics for my plots
def collect_statistics(actions_vs_episodes_all, final_performance_episodes):
    actions_taken_average = np.mean(actions_vs_episodes_all, 0)
    actions_taken_std = np.std(actions_vs_episodes_all, 0)

    final_performance_mean = np.mean(
        actions_taken_average[-final_performance_episodes:]
    )
    final_performance_std = np.mean(actions_taken_std[-final_performance_episodes:])

    return (
        actions_taken_average,
        actions_taken_std,
        final_performance_mean,
        final_performance_std,
    )
