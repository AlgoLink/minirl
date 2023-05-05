import gym
import numpy as np
from itertools import count
import traceback
from minirl.algos.a2c import A2C

env = gym.make("CartPole-v1")
ob_n = env.observation_space.shape[0]
ac_n = env.action_space.n

args = {}


def _2d_list(n):
    return [[] for _ in range(n)]


num_envs = 1

agent = A2C(ob_n, ac_n)
avg_reward = []
log_interval = 100
for i_episode in count(1):
    ep_reward = 0
    done = False
    state, _ = env.reset()
    state = np.expand_dims(state, axis=1)
    # Buffers to hold trajectories, e.g. `env_xs[i]` will hold the observations for environment `i`.
    mb_states = []
    mb_next_states = []
    mb_actions = []
    mb_action_probs = []
    mb_clipped_rewards = []
    mb_dones = []
    for t in range(10000):  # Don't infinite loop while learning
        while not done:
            mb_states.append(state)
            action, p = agent.act(state)

            mb_actions.append(action)

            state, reward, done, _, _ = env.step(action)
            state = np.expand_dims(state, axis=1)

            mb_action_probs.append(p)
            mb_next_states.append(state)
            mb_dones.append(done)
            mb_clipped_rewards.append(reward)

            ep_reward += reward

    try:
        agent.learn(
            states=mb_states,
            actions=mb_actions,
            action_probs=mb_action_probs,
            next_states=mb_next_states,
            rewards=mb_clipped_rewards,
            dones=mb_dones,
        )
        mb_states = []
        mb_next_states = []
        mb_actions = []
        mb_action_probs = []
        mb_dones = []
        mb_clipped_rewards = []
    except:
        print(str(traceback.format_exc()), len(mb_clipped_rewards))
        pass

    if i_episode % log_interval == 0 and i_episode > 0:
        print("Ave reward: {}".format(sum(avg_reward) / len(avg_reward)))
        avg_reward = []

    else:
        avg_reward.append(ep_reward)
