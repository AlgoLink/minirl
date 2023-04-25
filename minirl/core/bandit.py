"""
 multiarmed_bandit.py
 We solve the multi-armed bandit problem using a classical epsilon-greedy
 agent with reward-average sampling as the estimate to action-value Q.
 This algorithm follows closely with the notation of Sutton's RL textbook.
 We set up bandit arms with fixed probability distribution of success,
 and receive stochastic rewards from each arm of +1 for success,
 and 0 reward for failure.
 The incremental update rule action-value Q for each (action a, reward r):
   n += 1
   Q(a) <- Q(a) + 1/n * (r - Q(a))
 where:
   n = number of times action "a" was performed
   Q(a) = value estimate of action "a"
   r(a) = reward of sampling action bandit (bandit) "a"
 Derivation of the Q incremental update rule:
   Q_{n+1}(a)
   = 1/n * (r_1(a) + r_2(a) + ... + r_n(a))
   = 1/n * ((n-1) * Q_n(a) + r_n(a))
   = 1/n * (n * Q_n(a) + r_n(a) - Q_n(a))
   = Q_n(a) + 1/n * (r_n(a) - Q_n(a))
"""
import numpy as np
import math
import random
import pickle

from ..common.base import Policy


class BanditEEAgent:
    def __init__(self, nActions, eps=0.3):
        self.nActions = nActions
        self.eps = eps
        self.n = np.zeros(nActions, dtype=np.int)  # action counts n(a)
        self.Q = np.zeros(nActions, dtype=np.float)  # value Q(a)

    def learn(self, action, reward):
        # Update Q action-value given (action, reward)
        # self.n[action] += 1
        self.Q[action] += (1.0 / self.n[action]) * (reward - self.Q[action])

    def act(self):
        # Epsilon-greedy policy
        if np.random.random() < self.eps:  # explore
            action = np.random.randint(self.nActions)
        else:  # exploit
            action = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
        self.n[action] += 1
        return action


class EpsilonGreedy(Policy):
    r"""$\varepsilon$-greedy bandit policy.
    Performs arm selection by using an $\varepsilon$-greedy bandit strategy. An arm is selected at each
    step. The best arm is selected (1 - $\varepsilon$)% of the time.
    Selection bias is a common problem when using bandits. This bias can be mitigated by using
    burn-in phase. Each model is given the chance to learn during the first `burn_in` steps.
    Parameters
    ----------
    epsilon
        The probability of exploring.
    decay
        The decay rate of epsilon.
    reward_obj
        The reward object used to measure the performance of each arm. This can be a metric, a
        statistic, or a distribution.
    burn_in
        The number of steps to use for the burn-in phase. Each arm is given the chance to be pulled
        during the burn-in phase. This is useful to mitigate selection bias.
    seed
        Random number generator seed for reproducibility.
    Examples
    --------
    >>> import gym
    >>> from minirl import stats
    >>> env = gym.make(
    ...     'river_bandits/CandyCaneContest-v0'
    ... )
    >>> _ = env.reset(seed=42)
    >>> _ = env.action_space.seed(123)
    >>> policy = bandit.EpsilonGreedy(epsilon=0.9, seed=101)
    >>> metric = stats.Sum()
    >>> while True:
    ...     action = next(policy.pull(range(env.action_space.n)))
    ...     observation, reward, terminated, truncated, info = env.step(action)
    ...     policy = policy.update(action, reward)
    ...     metric = metric.update(reward)
    ...     if terminated or truncated:
    ...         break
    >>> metric
    Sum: 775.
    References
    ----------
    [^1]: [ε-Greedy Algorithm - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#%CE%B5-greedy-algorithm)
    """

    def __init__(
        self,
        epsilon: float,
        decay=0.0,
        n_actions=None,
        reward_obj=None,
        burn_in=0,
        seed: int = None,
    ):
        super().__init__(reward_obj, burn_in)
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.decay = decay
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def current_epsilon(self):
        """The value of epsilon after factoring in the decay rate."""
        if self.decay:
            return self.epsilon * math.exp(-self._n * self.decay)
        return self.epsilon

    def _pull(self, arm_ids):
        return (
            self._rng.choice(arm_ids)  # explore
            if self._rng.uniform(0, 1) < self.current_epsilon
            else max(arm_ids, key=lambda arm: self._rewards[arm].get())  # exploit
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"epsilon": 0.2}


class BanditRlite:
    def __init__(self, actions, eps=0, model_db=None, score_db=None):
        self._model_db = model_db
        self._score_db = score_db
        self._actions = actions
        self.q_model = {}

        self.eps = eps

    def learn(self, action, reward, model_id=None):
        # Update Q action-value given (action, reward)
        # self.n[action] += 1
        expose_key = f"{model_id}:{action}:BE"
        q_model = self.get_q_model(model_id)
        self.q_model = q_model
        _n = self._model_db.get(expose_key)
        if _n is None:
            n = 1
        else:
            n = max(1, float(_n))
        if action not in self.q_model:
            self.q_model[action]=0.0
        self.q_model[action] += (1.0 / n) * (reward - self.q_model[action])
        qvalue = self.q_model[action]
        self.update_q_model(self.q_model, model_id)
        self.update_q_score(qvalue, action, model_id)

    def update_q_score(self, qvalue, action, model_id):
        score_key = f"{model_id}:BQscore"

        if float(qvalue) > 0:
            Q_score = "-{}".format(qvalue)
        else:
            Q0_state_value = abs(qvalue)
            Q_score = "{}".format(Q0_state_value)

        self._score_db.zadd(score_key, Q_score, str(action))

    def update_q_model(self, Q_dict, model_id):
        model_key = f"{model_id}:Bqvalue"
        self._model_db.set(model_key, pickle.dumps(Q_dict))

        return model_key

    def get_q_model(self, model_id):
        model_key = f"{model_id}:Bqvalue"
        _model = self._model_db.get(model_key)
        if _model is None:
            model = {}
        else:
            model = pickle.loads(_model)
        return model

    def act(self, model_id, expose=True):
        # Epsilon-greedy policy
        if np.random.random() < self.eps:  # explore
            action = self.get_random_action(1)
        else:  # exploit
            action = self.greedy_action_selection(model_id)

        expose_key = f"{model_id}:{action}:BE"
        if expose:
            self._increment_customitem_tries(expose_key)
        return action

    # update custom itemkey cnt
    def _increment_customitem_tries(self, key: str) -> None:
        customitem_tries = self._model_db.incr(
            key
        )  # self.rlite_client.command("incr",key_tries)
        return customitem_tries

    def get_random_action(self, topN):
        if topN > len(self._actions):
            raise Exception("topN is longer than len of self.actions")
        return np.random.choice(
            self._actions, size=topN, replace=False, p=None
        ).tolist()

    def greedy_action_selection(self, model_id=None, topN=1, withscores=False):
        """
        Selects action with the highest Q-value for the given state.
        """
        # Get all the Q-values for all possible actions for the state
        maxQ_action_list = self.get_maxQ(model_id, topN, withscores=withscores)
        if len(maxQ_action_list) < 1:
            maxQ_action_list = self.get_random_action(topN)
        return maxQ_action_list

    def get_maxQ(self, model_id, topN, withscores=False):
        score_key = f"{model_id}:BQscore"
        if withscores:
            score_list = self._score_db.zrange(
                score_key, "0", str(topN - 1), "withscores"
            )
        else:
            score_list = self._score_db.zrange(score_key, "0", str(topN - 1))
        return score_list
