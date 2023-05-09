from itertools import count
import gym
import numpy as np
from minirl.agents.pg import PGAgent

from mlopskit import make

model_db = make("cache/feature_store-v1", db_name="lmodel.db")
model_id = "pg"
env = gym.make("CartPole-v1")
ob_n = env.observation_space.shape[0]
ac_n = env.action_space.n
agent = PGAgent(ob_n, ac_n, model_db=model_db, model_id=model_id)
render_interval = -1
log_interval = 100


def main():
    """Run REINFORCE algorithm to train on the environment"""
    avg_reward = []
    for i_episode in count(1):
        ep_reward = 0
        obs, _ = env.reset()
        for t in range(100):  # Don't infinite loop while learning

            action = agent.act(obs, model_id)
            next_obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            agent.rewards.append(reward)

            if render_interval != -1 and i_episode % render_interval == 0:
                env.render()

            if done:
                break

            obs = next_obs

        agent.learn(model_id)

        if i_episode % log_interval == 0:
            print("Ave reward: {}".format(sum(avg_reward) / len(avg_reward)))
            avg_reward = []

        else:
            avg_reward.append(ep_reward)


main()
