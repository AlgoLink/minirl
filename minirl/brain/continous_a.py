import numpy as np
import copy  # for deepcopy of model parameters
import traceback
import pickle


def gaussian(x, mean, std):
    """Gaussian probability density function. Book eq. 13.18"""
    var = np.power(std, 2)
    denom = (2 * np.pi * var) ** 0.5
    num = np.exp(-np.power(x - mean, 2) / (2 * var))
    return num / denom


class TabularGaussinaPolicy:
    """Tabular action-state function 'approximator'"""

    def __init__(self, lr, nb_states, nb_actions, l2=0.0, ent=0.1):
        assert isinstance(lr, float)
        assert isinstance(nb_states, tuple)
        assert isinstance(nb_actions, int)
        self._lr = lr  # learning rate
        self._l2 = l2  # L2 reg.
        self._ent = ent  # entropy reg.
        self.n_act = nb_actions
        self._theta_mu = np.zeros((*nb_states, nb_actions))  # weights
        self._theta_sigma = np.zeros((*nb_states, nb_actions))  # weights

    def pi(self, state, theta_mu, theta_sigma):
        """Return policy, i.e. probability distribution over actions."""
        _theta_mu = copy.deepcopy(theta_mu)
        _theta_sigma = copy.deepcopy(theta_sigma)
        assert isinstance(state, (int, tuple))
        assert _theta_mu.ndim == 2 if isinstance(state, int) else len(state) + 1
        assert _theta_sigma.ndim == 2 if isinstance(state, int) else len(state) + 1

        # Eq. 13.20
        # in tabular case x(s) vectors are one-hot, which is same as table lookup
        mu = _theta_mu[state].copy()
        sigma = np.exp(_theta_sigma[state])
        return mu, sigma  # do not sample here

    def update(self, state, action, theta_mu, theta_sigma, disc_return):
        _theta_mu = copy.deepcopy(theta_mu)
        _theta_sigma = copy.deepcopy(theta_sigma)
        assert isinstance(disc_return, float)

        mu = _theta_mu[state]  # Eq. 13.20
        sigma = np.exp(_theta_sigma[state])

        grad_ln_theta_mu = (1 / sigma**2) * (action - mu)  # Ex. 13.4
        grad_ln_theta_sigma = ((action - mu) ** 2 / sigma**2) - 1

        # L2 regularization - helps to ensure policy doesn't get deterministic
        grad_ln_theta_mu -= self._l2 * _theta_mu[state]
        grad_ln_theta_sigma -= self._l2 * _theta_sigma[state]

        # entropy reg. - also helps to ensure policy doesn't get deterministic
        prob = gaussian(action, mu, sigma)
        entropy = -1 * np.sum(prob * np.log(prob))
        grad_ln_theta_mu -= self._ent * entropy
        grad_ln_theta_sigma -= self._ent * entropy

        # apply update
        _theta_mu[state] += self._lr * grad_ln_theta_mu * disc_return
        _theta_sigma[state] += self._lr * grad_ln_theta_sigma * disc_return

        return _theta_mu, _theta_sigma


class ContinousAgent:
    """Implementation of <b>Policy for Continuous Actions</b> equations<"""

    def __init__(
        self, nb_states, nb_actions, model_db=None, lr=0.5, l2=0.0, ent=0.1
    ) -> None:
        self.nb_states = nb_states
        self.actions_n = nb_actions
        self.policy = TabularGaussinaPolicy(
            lr=lr, nb_states=nb_states, nb_actions=nb_actions, l2=l2, ent=ent
        )
        self._model_db = model_db

    def act(self, state, model_id=None):
        _theta_mu, _theta_sigma = self.load_weights(model_id=model_id)
        mu, sigma = self.policy.pi(
            state=state, theta_mu=_theta_mu, theta_sigma=_theta_sigma
        )
        action = np.random.normal(loc=mu, scale=sigma)

        return action

    def learn(self, state, action, reward, model_id=None):
        _theta_mu, _theta_sigma = self.load_weights(model_id=model_id)
        _theta_mu_updated, _theta_sigma_updated = self.policy.update(
            state=state,
            action=action,
            theta_mu=_theta_mu,
            theta_sigma=_theta_sigma,
            disc_return=reward,
        )
        self.save_weights(
            model_id=model_id,
            theta_mu=_theta_mu_updated,
            theta_sigma=_theta_sigma_updated,
        )

    def model_params_key(self, model_id):
        return f"{model_id}:cona"

    def save_weights(self, model_id, theta_mu, theta_sigma):
        if self._model_db is None:
            self.policy._theta_mu, self.policy._theta_sigma = theta_mu, theta_sigma
        else:
            model_key = self.model_params_key(model_id)
            self._model_db.set(model_key, pickle.dumps([theta_mu, theta_sigma]))

    def load_weights(self, model_id=None):
        try:
            if self._model_db is None:
                _theta_mu, _theta_sigma = (
                    self.policy._theta_mu,
                    self.policy._theta_sigma,
                )
            else:
                model_key = self.model_params_key(model_id)
                model = self._model_db.get(model_key)
                if model is not None:
                    _theta_mu, _theta_sigma = pickle.loads(model)
                    return _theta_mu, _theta_sigma
                else:
                    return self.policy._theta_mu, self.policy._theta_sigma

        except:
            print("Could not load weights: File Not Found, use default")
            print(traceback.format_exc())

            return self.policy._theta_mu, self.policy._theta_sigma
