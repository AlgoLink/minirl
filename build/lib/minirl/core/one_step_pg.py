"""
# ppo works by running the environment for some steps and then computing 
# stochastic gradient descent on the collected s,a,r,s' examples.
# I use a value function for the baseline just like in actor-critic
"""
import numpy as np
from ..common.pg_models import ValueFunction, Policy
from ..common.replay_buffer import ReplayMemory
from river.preprocessing import StandardScaler

gamma = 0.99


def discount_rewards(r):
    """take 1D float array of rewards and compute discounted reward"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = (
                0  # reset the sum, since this was a game boundary (pong specific!)
            )
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        # discounted_r = (discounted_r - discounted_r.mean()) / (
        #        discounted_r.std() + np.finfo(np.float32).eps
        #    )
    return discounted_r


class uniAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        p_alpha=0.1,
        v_alpha=0.01,
        gamma=0.95,
        algo="ppo",
        clip=None,
        capacity=20,
        batch_size=7,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.p_alpha = p_alpha
        self.v_alpha = v_alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.reward_process = StandardScaler()
        VALID_ALGOS = ("ppo", "ac")
        assert (
            algo in VALID_ALGOS
        ), f"rl algo method: {algo} not supported. Valid model service status are {VALID_ALGOS}"
        self.algo = algo
        print(f"simple one step {algo} neural network")
        n_hidden = 64
        print(
            f"creating nn: #input:{state_dim} #hidden:{n_hidden} #output:{action_dim}"
        )
        if algo == "ppo":
            self.policy = Policy(self.p_alpha, state_dim, action_dim, clip=clip)
        else:
            self.policy = Policy(self.p_alpha, state_dim, action_dim, clip=None)

        self.value_function = ValueFunction(self.v_alpha, state_dim)
        self.capacity = capacity
        self.replay_buffer = ReplayMemory(capacity=capacity)
        self.temp_replay_buffer = ReplayMemory(capacity=1)

    def act(self, state, save_his=True):
        if self.algo == "ppo":
            # draw action from policy
            action, action_probabilities = self.policy.get_action(state, False)
            if save_his:
                self.temp_replay_buffer.push([state, action, action_probabilities])

        else:
            # draw action from policy
            action, action_probabilities = self.policy.get_action(state, True)

        return action, action_probabilities

    def learn(self, state, reward, next_state):
        if self.algo == "ac":
            self.ac_update(state, reward, next_state)
        else:
            self.ppo_update(reward, next_state)

        return True

    def ac_update(self, state, reward, next_state):
        # compute TD error
        next_state_value = self.value_function.evaluate_state(next_state)
        state_value = self.value_function.evaluate_state(state)
        target = reward + self.gamma * next_state_value
        td_error = target - state_value
        # update actor & critic
        self.policy.update_parameters(td_error, state)
        self.value_function.update_parameters(td_error, state)

    def ppo_update(self, reward, next_state):
        (
            state_features,
            action_probabilities,
            actions,
            targets,
            advantages,
        ) = self.proximal_policy_optimization(reward, next_state)
        # print(state_features,"state_features")
        if state_features is not None:
            self.policy.ppo_gradient_step(
                state_features, action_probabilities, actions, advantages
            )
            state_values = self.value_function.evaluate_state(state_features)
            errors = targets - state_values
            self.value_function.update_parameters(errors, state_features)
            # self.replay_buffer = ReplayMemory(capacity=self.capacity)

    def proximal_policy_optimization(self, reward, next_state):
        # run the environment
        tmp_buffer = self.ppo_rollout(reward, next_state)
        self.replay_buffer.push(tmp_buffer)
        state_features, action_probabilities, actions, targets, advantages = (
            None,
            None,
            None,
            None,
            None,
        )
        if self.replay_buffer.memory.qsize() > self.batch_size:
            samples = self.replay_buffer.sample(self.batch_size)
            state_features = []
            next_state_features = []
            actions = []
            action_probabilities = []
            rewards = []

            for sample in samples:
                state, action, _action_probabilities, reward, next_state = sample
                state_features.append(state)
                actions.append(action)
                # x={"reward":reward}
                # _r = self.reward_process.learn_one(x).transform_one(x)
                # rewards.append(_r["reward"])
                rewards.append(reward)
                action_probabilities.append(_action_probabilities)
                next_state_features.append(next_state)
            state_features = np.array(state_features)
            # print(state_features,"state_features")
            actions = np.array(actions)
            action_probabilities = np.array(action_probabilities)
            # r = self.calculate_discounted_returns(rewards)
            rewards = np.array(rewards)
            # discounted_epr=discount_rewards(np.array(rewards,dtype=np.float64))
            # discounted_epr -= discounted_epr.mean()
            # discounted_epr /= (discounted_epr.std()+ np.finfo(np.float32).eps)
            # rewards = discounted_epr
            state_values = np.squeeze(
                self.value_function.evaluate_state(state_features)
            )
            next_state_values = np.squeeze(
                self.value_function.evaluate_state(next_state_features)
            )
            targets = (
                rewards + self.gamma * next_state_values
            )  # super easy td error calculation
            advantages = targets - state_values

        return state_features, action_probabilities, actions, targets, advantages

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
        # normalize for better statistical properties
        returns = (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps
        )
        return returns

    def ppo_rollout(self, reward, next_state):

        if self.temp_replay_buffer.memory.qsize() > 0:
            tmp_buffer = self.temp_replay_buffer.memory.queue[0]
            tmp_buffer.append(reward)
            tmp_buffer.append(next_state)
            # state = tmp_buffer[0]
            # state_values = self.value_function.evaluate_state(state)
            # next_state_values = self.value_function.evaluate_state(next_state)
            # targets = reward + self.gamma * next_state_values # super easy td error calculation
            # advantages = targets - state_values
            # tmp_buffer.append()
        return tmp_buffer
