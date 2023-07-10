"""LinUCB with Disjoint Linear Models

This module contains a class that implements LinUCB with disjoint linear model,
a contextual bandit algorithm assuming the reward function is a linear function
of the context.
"""
import logging

import six
import numpy as np
import copy
import pickle

LOGGER = logging.getLogger(__name__)


class LinUCB:
    r"""LinUCB with Disjoint Linear Models

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    recommendation_cls : class (default: None)
        The class used to initiate the recommendations. If None, then use
        default Recommendation class.

    context_dimension: int
        The dimension of the context.

    alpha: float
        The constant determines the width of the upper confidence bound.

    References
    ----------
    .. [1]  Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
            News Article Recommendation." In Proceedings of the 19th
            International Conference on World Wide Web (WWW), 2010.
    """

    def __init__(self, actions, obs_dim=128, alpha=0.5, model_db=None):

        self.alpha = alpha
        self.obs_dim = obs_dim
        self.actions = actions
        self._model_db = model_db

        self._init_model()

    def _init_model(self):

        # Initialize LinUCB Model Parameters
        self.model = {
            # dictionary - For any action a in actions,
            # A[a] = (DaT*Da + I) the ridge reg solution
            "A": {},
            # dictionary - The inverse of each A[a] for action a
            # in actions
            "A_inv": {},
            # dictionary - The cumulative return of action a, given the
            # context xt.
            "b": {},
            # dictionary - The coefficient vector of actiona with
            # linear model b = dot(xt, theta)
            "theta": {},
        }

        for action_id in self.actions:
            self._init_action_model(action_id)

    def _init_action_model(self, action_id):
        self.model["A"][action_id] = np.identity(self.obs_dim)
        self.model["A_inv"][action_id] = np.identity(self.obs_dim)
        self.model["b"][action_id] = np.zeros((self.obs_dim, 1))
        self.model["theta"][action_id] = np.zeros((self.obs_dim, 1))

    def _linucb_score(self, context, model):
        """disjoint LINUCB algorithm."""
        A_inv = model["A_inv"]  # pylint: disable=invalid-name
        theta = model["theta"]

        # The recommended actions should maximize the Linear UCB.
        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id in self.actions:
            action_context = np.reshape(context[action_id], (-1, 1))
            estimated_reward[action_id] = float(theta[action_id].T.dot(action_context))
            uncertainty[action_id] = float(
                self.alpha
                * np.sqrt(action_context.T.dot(A_inv[action_id]).dot(action_context))
            )
            score[action_id] = estimated_reward[action_id] + uncertainty[action_id]
        return estimated_reward, uncertainty, score

    def act(self, context, model_id, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dict
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if not isinstance(context, dict):
            raise ValueError("LinUCB requires context dict for all actions!")

        model = self.load_weights(model_id)
        estimated_reward, uncertainty, score = self._linucb_score(context, model)

        if n_actions is None:
            recommendation_id = max(score, key=score.get)
            action = self.actions[recommendation_id]

        else:
            recommendation_ids = sorted(score, key=score.get, reverse=True)[:n_actions]
            recommendations = []  # pylint: disable=redefined-variable-type
            for action_id in recommendation_ids:
                recommendations.append(self.actions[action_id])

            action = recommendations

        return action

    def learn(self, context, rewards, model_id):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        # Update the model
        model = self.load_weights(model_id)
        A = model["A"]  # pylint: disable=invalid-name
        A_inv = model["A_inv"]  # pylint: disable=invalid-name
        b = model["b"]
        theta = model["theta"]

        for action_id, reward in six.viewitems(rewards):
            action_context = np.reshape(context[action_id], (-1, 1))
            A[action_id] += action_context.dot(action_context.T)
            A_inv[action_id] = np.linalg.inv(A[action_id])
            b[action_id] += reward * action_context
            theta[action_id] = A_inv[action_id].dot(b[action_id])
        self.save_weights(
            model_id,
            {
                "A": A,
                "A_inv": A_inv,
                "b": b,
                "theta": theta,
            },
        )

    def get_model_key(self, model_id):
        return f"{model_id}:linucb"

    def set_weights(self, model):
        self.model = copy.deepcopy(model)

    def load_weights(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            model = self.model
        else:
            model = pickle.loads(_model)
        return model

    def save_weights(self, model_id, model):
        model_key = self.get_model_key(model_id)
        self._model_db.set(model_key, pickle.dumps(model))

    def add_action(self, actions):
        """Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        new_action_ids = self._action_storage.add(actions)
        model = self._model_storage.get_model()

        for action_id in new_action_ids:
            self._init_action_model(model, action_id)

        self._model_storage.save_model(model)

    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        model = self._model_storage.get_model()
        del model["A"][action_id]
        del model["A_inv"][action_id]
        del model["b"][action_id]
        del model["theta"][action_id]
        self._model_storage.save_model(model)
        self._action_storage.remove(action_id)
