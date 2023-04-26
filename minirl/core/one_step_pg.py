"""
# ppo works by running the environment for some steps and then computing 
# stochastic gradient descent on the collected s,a,r,s' examples.
# I use a value function for the baseline just like in actor-critic
"""
import numpy as np
from ..common.pg_models import ValueFunction, Policy
from ..common.replay_buffer import ReplayMemory


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
        capacity=7,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.p_alpha = p_alpha
        self.v_alpha = v_alpha
        self.gamma = gamma
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
        self.replay_buffer = ReplayMemory(capacity=capacity)
        self.temp_replay_buffer = ReplayMemory(capacity=1)

    def act(self, state,save_his=True):
        if self.algo == "ppo":
            # draw action from policy
            action, action_probabilities = self.policy.get_action(state, False)

        else:
            # draw action from policy
            action, action_probabilities = self.policy.get_action(state, True)
            if save_his:
                self.temp_replay_buffer.push([state,action])

        return action, action_probabilities

    def learn(self, state, reward, next_state):
        if self.algo == "ac":
            self.one_step_actor_critic(state, reward, next_state)

        return True

    def one_step_actor_critic(self, state, reward, next_state):
        # compute TD error
        next_state_value = self.value_function.evaluate_state(next_state)
        state_value = self.value_function.evaluate_state(state)
        target = reward + self.gamma * next_state_value
        td_error = target - state_value
        # update actor & critic
        self.policy.update_parameters(td_error, state)
        self.value_function.update_parameters(td_error, state)

    def proximal_policy_optimization(self):
        # run the environment
        (
            state_features,
            actions,
            action_probabilities,
            targets,
            advantages,
            episode_lengths,
        ) = self.ppo_rollout(
            mdp, policy, value_function, current_rollout_episodes, max_actions
        )

        def ppo_rollout(self, policy, value_function, rollout_episodes, max_actions):
            state_features = []

            actions = []
            action_probabilities = []
            rewards = []

            last_state_features = []
            episode_lengths = []
