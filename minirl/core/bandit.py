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
"""A module containing different variations on multi-armed bandit environments."""

from abc import ABC, abstractmethod

import numpy as np
import math
import random
import pickle

from ..common.base import Policy
from ..common.testing import is_number


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
            self.q_model[action] = 0.0
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


# bandit for linucb
class Bandit(ABC):
    def __init__(self, rewards, reward_probs, context=None):
        assert len(rewards) == len(reward_probs)
        self.step = 0
        self.n_arms = len(rewards)

        super().__init__()

    def __repr__(self):
        """A string representation for the bandit"""
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {}

    @abstractmethod
    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms, if
            applicable. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        """
        pass

    def pull(self, arm_id, context=None):
        """
        "Pull" (i.e., sample from) a given arm's payoff distribution.

        Parameters
        ----------
        arm_id : int
            The integer ID of the arm to sample from
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D,)` or None
            The context vector for the current timestep if this is a contextual
            bandit. Otherwise, this argument is unused and defaults to None.

        Returns
        -------
        reward : float
            The reward sampled from the given arm's payoff distribution
        """
        assert arm_id < self.n_arms

        self.step += 1
        return self._pull(arm_id, context)

    def reset(self):
        """Reset the bandit step and action counters to zero."""
        self.step = 0

    @abstractmethod
    def _pull(self, arm_id):
        pass


class ContextualLinearBandit(Bandit):
    def __init__(self, K, D, payoff_variance=1):
        r"""
        A contextual linear multi-armed bandit.

        Notes
        -----
        In a contextual linear bandit the expected payoff of an arm :math:`a
        \in \mathcal{A}` at time `t` is a linear combination of its context
        vector :math:`\mathbf{x}_{t,a}` with a coefficient vector
        :math:`\theta_a`:

        .. math::

            \mathbb{E}[r_{t, a} \mid \mathbf{x}_{t, a}] = \mathbf{x}_{t,a}^\top \theta_a

        In this implementation, the arm coefficient vectors :math:`\theta` are
        initialized independently from a uniform distribution on the interval
        [-1, 1], and the specific reward at timestep `t` is normally
        distributed:

        .. math::

            r_{t, a} \mid \mathbf{x}_{t, a} \sim
                \mathcal{N}(\mathbf{x}_{t,a}^\top \theta_a, \sigma_a^2)

        Parameters
        ----------
        K : int
            The number of bandit arms
        D : int
            The dimensionality of the context vectors
        payoff_variance : float or :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
            The variance of the random noise in the arm payoffs. If a float,
            the variance is assumed to be equal for each arm. Default is 1.
        """
        if is_number(payoff_variance):
            payoff_variance = [payoff_variance] * K

        assert len(payoff_variance) == K
        assert all(v > 0 for v in payoff_variance)

        self.K = K
        self.D = D
        self.payoff_variance = payoff_variance

        # use a dummy placeholder variable to initialize the Bandit superclass
        placeholder = [None] * K
        super().__init__(placeholder, placeholder)

        # initialize the theta matrix
        self.thetas = np.random.uniform(-1, 1, size=(D, K))
        self.thetas /= np.linalg.norm(self.thetas, 2)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {
            "id": "ContextualLinearBandit",
            "K": self.K,
            "D": self.D,
            "payoff_variance": self.payoff_variance,
        }

    @property
    def parameters(self):
        """A dictionary of the current bandit parameters"""
        return {"thetas": self.thetas}

    def get_context(self):
        """
        Sample the context vectors for each arm from a multivariate standard
        normal distribution.

        Returns
        -------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)`
            A `D`-dimensional context vector sampled from a standard normal
            distribution for each of the `K` bandit arms.
        """
        return np.random.normal(size=(self.D, self.K))

    def oracle_payoff(self, context):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms, if
            applicable. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        best_arm = np.argmax(self.arm_evs)
        return self.arm_evs[best_arm], best_arm

    def _pull(self, arm_id, context):
        K, thetas = self.K, self.thetas
        self._noise = np.random.normal(scale=self.payoff_variance, size=self.K)
        self.arm_evs = np.array([context[:, k] @ thetas[:, k] for k in range(K)])
        return (self.arm_evs + self._noise)[arm_id]
