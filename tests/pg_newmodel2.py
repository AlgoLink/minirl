from itertools import count
import gym
import numpy as np
from minirl.agents.pg import PGAgent
import traceback
from mlopskit import make

model_db = make("cache/feature_store-v1", db_name="lmodel2.db")
score_db = make("cache/feature_store-v1", db_name="lmodel3.db")
model_id = "pg4"
env = gym.make("CartPole-v1")
ob_n = env.observation_space.shape[0]
ac_n = env.action_space.n
agent = PGAgent(
    ob_n, ac_n, model_db=model_db, score_db=score_db, his_db=score_db, model_id=model_id
)
render_interval = -1
log_interval = 100


def main():
    """Run REINFORCE algorithm to train on the environment"""
    avg_reward = []
    for i_episode in count(1):
        ep_reward = 0
        obs, _ = env.reset()
        for t in range(100):  # Don't infinite loop while learning

            action = agent.act(obs, model_id, save_aprobs=False)
            next_obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if reward < 0 or reward > 1:
                print(reward, "rewaed")
            # agent.rewards.append(reward)
            # agent.policy._add_to_cache_using_rpush("rewards", str(reward), model_id)
            # reward_local_key = agent.policy.cache_local_key("rewards", model_id)

            if render_interval != -1 and i_episode % render_interval == 0:
                env.render()

            if done:
                break

            obs = next_obs
        # agent.policy._score_db.delete(reward_local_key)
        try:
            x1 = 1
            # agent.learn(model_id)
            # print(traceback.format_exc())
            # x=1
        except:
            reward_local_key = agent.policy.cache_local_key("rewards", model_id)
            aprobs_local_key = agent.policy.cache_local_key("aprobs", model_id)
            agent.policy._score_db.delete(reward_local_key)
            agent.policy._score_db.delete(aprobs_local_key)
            cache_key = agent.policy.cache_key(model_id)
            agent.policy._score_db.delete(cache_key)

        if i_episode % log_interval == 0:
            print("Ave reward: {}".format(sum(avg_reward) / len(avg_reward)))
            avg_reward = []

        else:
            avg_reward.append(ep_reward)


main()
