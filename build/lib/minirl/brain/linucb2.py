from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pickle
import copy  # for deepcopy of model parameters


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

    def act(self, arms, context=None):
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
            self._initialize_params(arms)

        arm_id = self._select_arm(arms, context)
        rwd = self._pull_arm(arms, arm_id, context)
        self._update_params(arm_id, rwd, context)
        return rwd, arm_id

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates, "A": self.A, "b": self.b}

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
    def __init__(self, obs_dim, act_dim, model_db=None, alpha=1, gamma=0.1):
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
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.gamma = gamma
        self._model_db = model_db
        self.A, self.b = [], []
        self.is_initialized = False

        if not self.is_initialized:
            self._initialize_params()

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

    def _initialize_params(self):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        bhp = self.hyperparameters
        fstr = "LinUCB only defined for contextual linear bandits, got: {}"
        assert bhp["id"] == "LinUCB", fstr.format(bhp["id"])

        H = 1
        # H:hidden_dim
        theta_init = (-1 + 2 * np.random.rand(self.obs_dim, H)) / np.sqrt(self.obs_dim)

        self.A, self.b = [], []
        for _ in range(self.act_dim):
            self.A.append(np.eye(self.obs_dim))
            self.b.append(np.zeros(self.obs_dim))
        self.theta = theta_init
        self.is_initialized = True

    def act(self, arms, model_id):
        self.set_weights(*self.get_weights(model_id))
        arm_id = self._select_arm(arms)
        return arm_id

    def _select_arm(self, arms):
        probs = []
        for i, x in enumerate(arms):
            C, A, b = x, self.A[i], self.b[i]
            A_inv = np.linalg.inv(A)
            theta_hat = A_inv @ b
            p = theta_hat @ C + self.alpha * np.sqrt(C.T @ A_inv @ C)

            probs.append(p)
        return np.argmax(probs)

    def learn(self, arm_id, r, xi, model_id=None):
        self.set_weights(*self.get_weights(model_id))
        self._update_params(arm_id, r, xi)
        self.save_weights(model_id)

    def _update_params(self, arm_id, r, xi):
        """Compute the parameters for A and b."""
        self.A[arm_id] += xi @ xi.T
        self.b[arm_id] += r * xi

    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        self.A, self.b = [], []
        self.ev_estimates = {}

    def get_model_key(self, model_id):
        return f"{model_id}:params"

    def get_weights(self, model_id):
        # return self.weights, self.biases
        A, b, theta = self.load_weights(model_id)
        return A, b, theta

        # return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))

    def set_weights(self, A, b, theta):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.A = copy.deepcopy(A)
        self.b = copy.deepcopy(b)
        self.theta = copy.deepcopy(theta)

    def save_weights(self, model_id):
        if self._model_db is None:
            pickle.dump(
                [self.A, self.b, self.theta],
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.get_model_key(model_id)
            self._model_db.set(model_key, pickle.dumps([self.A, self.b, self.theta]))

    def load_weights(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            A, b, theta = self.A, self.b, self.theta
        else:
            A, b, theta = pickle.loads(_model)
        return A, b, theta
