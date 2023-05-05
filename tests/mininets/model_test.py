import model
import gym
import numpy as np
from itertools import count
from config import Config

env = gym.make("CartPole-v1")
ob_n = env.observation_space.shape[0]
ac_n = env.action_space.n

args = {}
config = Config(args)


def _2d_list(n):
    return [[] for _ in range(n)]


num_envs = 1

agent = model.Agent(ob_n, ac_n, config=config)
avg_reward = []
log_interval = 100
for i_episode in count(1):
    ep_reward = 0
    obs, _ = env.reset()

    # Buffers to hold trajectories, e.g. `env_xs[i]` will hold the observations for environment `i`.
    env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
    env_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)
    for t in range(10000):  # Don't infinite loop while learning
        step_as, step_vs = agent.act([obs])
        _obs = [obs]
        step_xs = np.vstack([np.array(o).ravel() for o in _obs])
        next_obs, reward, done, _, _ = env.step(step_as[0])
        ep_reward += reward
        # Record the observation, action, value, and reward in the buffers.
        env_xs[0].append(step_xs[0].ravel())
        env_as[0].append(step_as[0])
        env_vs[0].append(step_vs[0][0])
        env_rs[0].append(reward)
        # print(t,"act")
        # if t%100==0 and t>0:
        # _, extra_vs = agent.forward(np.vstack(observations).reshape(num_envs, -1))
        # Perform update and clear buffers.
        # print(env_xs,t,"learn")
        # env_xs = np.vstack(env_xs).reshape(len(env_xs),-1)
        # env_as = np.vstack(env_as).reshape(len(env_as),-1)
        # env_rs = np.vstack(env_rs).reshape(len(env_rs),-1)
        # env_vs = np.vstack(env_xs).reshape(len(env_vs),-1)

        # agent.train_step(env_xs, env_as, env_rs, env_vs)
        # env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
        # env_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)

        if done:
            env_vs[0].append(0.0)
            break
        else:
            obs = next_obs

    try:
        agent.train_step(env_xs, env_as, env_rs, env_vs)
        env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
        senv_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)
    except:
        pass

    if i_episode % log_interval == 0 and i_episode > 0:
        print("Ave reward: {}".format(sum(avg_reward) / len(avg_reward)))
        avg_reward = []

    else:
        avg_reward.append(ep_reward)
