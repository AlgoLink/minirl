"""A module containing exploration policies for various multi-armed bandit problems."""

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class BanditPolicyBase(ABC):
    def __init__(self):
        """A simple base class for multi-armed bandit policies"""
        self.step = 0
        self.ev_estimates = {}
        self.is_initialized = False
        super().__init__()

    def __repr__(self):
        """Return a string representation of the policy"""
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        pass

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        pass

    def act(self, bandit, context=None):
        """
        Select an arm and sample from its payoff distribution.

        Parameters
        ----------
        bandit : :class:`Bandit <numpy_ml.bandits.bandits.Bandit>` instance
            The multi-armed bandit to act upon
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D,)` or None
            The context vector for the current timestep if interacting with a
            contextual bandit. Otherwise, this argument is unused. Default is
            None.

        Returns
        -------
        rwd : float
            The reward received after pulling ``arm_id``.
        arm_id : int
            The arm that was pulled to generate ``rwd``.
        """
        if not self.is_initialized:
            self._initialize_params(bandit)

        arm_id = self._select_arm(bandit, context)
        rwd = self._pull_arm(bandit, arm_id, context)
        self._update_params(arm_id, rwd, context)
        return rwd, arm_id

    def reset(self):
        """Reset the policy parameters and counters to their initial states."""
        self.step = 0
        self._reset_params()
        self.is_initialized = False

    def _pull_arm(self, bandit, arm_id, context):
        """Execute a bandit action and return the received reward."""
        self.step += 1
        return bandit.pull(arm_id, context)

    @abstractmethod
    def _select_arm(self, bandit, context):
        """Select an arm based on the current context"""
        pass

    @abstractmethod
    def _update_params(self, bandit, context):
        """Update the policy parameters after an interaction"""
        pass

    @abstractmethod
    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        pass

    @abstractmethod
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        pass


class LinUCB(BanditPolicyBase):
    def __init__(self, alpha=1):
        """
        A disjoint linear UCB policy [*]_ for contextual linear bandits.

        Notes
        -----
        LinUCB is only defined for :class:`ContextualLinearBandit <numpy_ml.bandits.ContextualLinearBandit>` environments.

        References
        ----------
        .. [*] Li, L., Chu, W., Langford, J., & Schapire, R. (2010). A
           contextual-bandit approach to personalized news article
           recommendation. In *Proceedings of the 19th International Conference
           on World Wide Web*, 661-670.

        Parameters
        ----------
        alpha : float
            A confidence/optimisim parameter affecting the amount of
            exploration. Default is 1.
        """  # noqa
        super().__init__()

        self.alpha = alpha
        self.A, self.b = [], []
        self.is_initialized = False

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates, "A": self.A, "b": self.b}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "LinUCB",
            "alpha": self.alpha,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        bhp = bandit.hyperparameters
        fstr = "LinUCB only defined for contextual linear bandits, got: {}"
        assert bhp["id"] == "ContextualLinearBandit", fstr.format(bhp["id"])

        self.A, self.b = [], []
        for _ in range(bandit.n_arms):
            self.A.append(np.eye(bandit.D))
            self.b.append(np.zeros(bandit.D))

        self.is_initialized = True

    def _select_arm(self, bandit, context):
        probs = []
        for a in range(bandit.n_arms):
            C, A, b = context[:, a], self.A[a], self.b[a]
            A_inv = np.linalg.inv(A)
            theta_hat = A_inv @ b
            p = theta_hat @ C + self.alpha * np.sqrt(C.T @ A_inv @ C)

            probs.append(p)
        return np.argmax(probs)

    def _update_params(self, arm_id, rwd, context):
        """Compute the parameters for A and b."""
        self.A[arm_id] += context[:, arm_id] @ context[:, arm_id].T
        self.b[arm_id] += rwd * context[:, arm_id]

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.A, self.b = [], []
        self.ev_estimates = {}
