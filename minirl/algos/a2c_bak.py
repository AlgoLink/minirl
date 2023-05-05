from ..models import activation, loss, optim
from ..models.nn import NeuralNetwork

import numpy as np


class A2C:
    def __init__(
        self,
        obs_n=None,
        ac_n=None,
        hidden_layers=[32],
        lr=1e-4,
        lr_c=1e-3,
        gamma=0.99,
        batch_size=7,
    ):
        self.lr_policy = lr
        self.lr_value = lr_c
        self.num_actions = ac_n
        self.num_states = obs_n
        self.gamma = gamma

        self.policy = NeuralNetwork(
            [self.num_states, *hidden_layers, self.num_actions],
            [*len(hidden_layers) * [activation.relu], activation.softmax],
        )

        self.policy_optimiser = optim.Momentum(self.policy, batch_size=batch_size)

        self.value = NeuralNetwork(
            [self.num_states, *hidden_layers, 1],
            [*len(hidden_layers) * [activation.relu], activation.linear],
        )
        self.value_optimiser = optim.Momentum(
            self.value, loss_fcn=loss.mean_squared_error, batch_size=batch_size
        )

        self.models = [self.policy, self.value]

    def act(self, state):
        action_prob = np.squeeze(self.policy.predict(state))
        action = np.random.choice(self.num_actions, p=action_prob)

        return action, action_prob

    def learn(self, states, actions, action_probs, next_states, rewards, dones):
        mb_states = np.hstack(states)
        mb_next_states = np.hstack(next_states)

        values = self.value.predict(mb_states)
        next_values = self.value.predict(mb_next_states)

        init_gradient = loss.cross_entropy_loss_deriv(
            np.array(actions), np.vstack(action_probs).T
        )

        mb_clipped_rewards = self.discount_rewards(rewards)
        mb_targets = np.array(mb_clipped_rewards) + self.gamma * next_values * (
            1 - np.array(dones)
        )
        td_errors = mb_targets - values

        init_gradient = init_gradient * td_errors
        self.value_optimiser.train(
            mb_states, mb_targets, learning_rate=self.lr_value / len(actions)
        )

        self.policy_optimiser.train(
            mb_states,
            np.array(actions),
            learning_rate=self.lr_policy / len(actions),
            init_gradient=init_gradient,
        )

    def discount_rewards(self, rewards, gamma):
        """take 1D float array of rewards and compute discounted reward"""
        r = rewards
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = (
                    0  # reset the sum, since this was a game boundary (pong specific!)
                )
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add

        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r


def A2C(args):

    # Initialise Environment
    env = gym.make(args.env_name)
    #    env.seed(0) # TODO
    #    np.random.seed(0) #TODO

    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    # Create Neural Networks
    lr_policy = args.lr
    lr_value = args.lr_c

    policy = NeuralNetwork(
        [num_states, *args.hidden_layers, num_actions],
        [*len(args.hidden_layers) * [activation.relu], activation.softmax],
    )

    policy_optimiser = optim.Momentum(policy, batch_size=args.batch_size)

    value = NeuralNetwork(
        [num_states, *args.hidden_layers, 1],
        [*len(args.hidden_layers) * [activation.relu], activation.linear],
    )

    value_optimiser = optim.Momentum(
        value, loss_fcn=loss.mean_squared_error, batch_size=args.batch_size
    )

    args.models = [policy, value]

    mb_states = []
    mb_next_states = []
    mb_actions = []
    mb_action_probs = []
    mb_clipped_rewards = []
    mb_dones = []
    #    gamma_discount = []

    episode_num = 0
    while (
        episode_num < args.num_episodes
    ):  # for loop does not handle changing of num_episodes
        episode_num += 1

        done = False
        state = env.reset()
        state = np.expand_dims(state, axis=1)

        ep_reward = 0

        while not done:
            if args.render:
                env.render()
            mb_states.append(state)

            action_prob = np.squeeze(policy.predict(state))
            action = np.random.choice(num_actions, p=action_prob)

            mb_actions.append(action)

            state, reward, done, _ = env.step(action)
            state = np.expand_dims(state, axis=1)

            #            rewards.append(reward)
            mb_action_probs.append(action_prob)
            # TODO
            #            gamma_discount.append(gamma**i)
            mb_next_states.append(state)
            mb_dones.append(done)

            #            if reward < -1:
            #                clipped_rewards.append(-1)
            #            elif reward >= 100:
            #                clipped_rewards.append(1)
            #            elif reward > 1:
            #                clipped_rewards.append(0.5)
            #            else:
            mb_clipped_rewards.append(reward)

            ep_reward += reward

        # Moving average of the reward
        args.mean_reward_deque.append(ep_reward)
        mean_reward = np.mean(args.mean_reward_deque)

        print(
            "({}) mean reward {:5.2f} reward: {:5.2f}".format(
                episode_num, mean_reward, ep_reward
            )
        )

        if args.log_dir is not None:
            args.reward_logger.append([ep_reward])
            if episode_num % 50 == 0:
                args.reward_logger.save()

        #    rewards = np.clip(rewards, -1, 1)

        # Update Policy and Baseline Networks
        if episode_num % args.episode_batch_size == 0:
            mb_states = np.hstack(mb_states)
            mb_next_states = np.hstack(mb_next_states)

            values = value.predict(mb_states)
            next_values = value.predict(mb_next_states)

            init_gradient = loss.cross_entropy_loss_deriv(
                np.array(mb_actions), np.vstack(mb_action_probs).T
            )

            mb_targets = np.array(mb_clipped_rewards) + args.gamma * next_values * (
                1 - np.array(mb_dones)
            )
            td_errors = mb_targets - values

            init_gradient = init_gradient * td_errors
            value_optimiser.train(
                mb_states, mb_targets, learning_rate=lr_value / len(mb_actions)
            )

            policy_optimiser.train(
                mb_states,
                np.array(mb_actions),
                learning_rate=lr_policy / len(mb_actions),
                init_gradient=init_gradient,
            )

            mb_states = []
            mb_next_states = []
            mb_actions = []
            mb_action_probs = []
            mb_dones = []
            mb_clipped_rewards = []

    env.close()
