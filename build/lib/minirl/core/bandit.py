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
