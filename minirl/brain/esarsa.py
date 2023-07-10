import numpy as np
import random
import copy  # for deepcopy of model parameters
import pickle
import traceback


def get_pareto_front(p):
    """
    Create Pareto optimal set of actions (Pareto front) A* identified as actions that are not dominated by any action
    out of the set A*.

    Parameters:
    -----------
    p: Dict[ActionId, Probability]
        The dictionary or actions and their sampled probability of getting a positive reward for each objective.

    Return
    ------
    pareto_front: set
        The list of Pareto optimal actions
        ex. ['a0', 'a1', 'a4', 'a5']
    """
    # store non dominated actions
    pareto_front = []

    for this_action in p.keys():
        is_pareto = (
            True  # we assume that action is Pareto Optimal until proven otherwise
        )
        other_actions = [a for a in p.keys() if a != this_action]

        for other_action in other_actions:
            # check if this_action is not dominated by other_action based on
            # multiple objectives reward prob vectors
            is_dominated = not (
                # an action cannot be dominated by an identical one
                (p[this_action] == p[other_action])
                # otherwise, apply the classical definition
                or any(
                    p[this_action][i] > p[other_action][i]
                    for i in range(len(p[this_action]))
                )
            )

            if is_dominated:
                # this_action dominated by at least one other_action,
                # this_action is not pareto optimal
                is_pareto = False
                break

        if is_pareto:
            # this_action is pareto optimal
            pareto_front.append(this_action)

    return pareto_front


class ExpectedSarsa:
    """
    Expected sarsa algorithm. Finds the optimal greedy policy

    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Probability of picking non-greedy action.
        extra_action: If set to True there are 8 possible actions an agent can take, if set to False it will be 4.
        add_stochasticity: If set to True stochasticity will be added, if set to False it won't be.

    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
        iteration_time_list which is an list containing the time it took to run each iteration.
    """

    def __init__(
        self, nb_states, nb_actions, eps=0.2, alpha=0.4, gamma=0.99, model_db=None
    ):
        assert isinstance(nb_states, tuple)
        assert isinstance(nb_actions, int)

        self.actions_n = nb_actions
        self.actions = [i for i in range(nb_actions)]
        self.nb_states = nb_states
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self._model_db = model_db

        self._init_model()

    def _init_model(self):
        self.Q = np.zeros((*self.nb_states, self.actions_n))

    def act(self, state, model_id, allowd_actions=None):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state
            extra_actions: If set to True choose out of 8 actions, 4 actions otherwise.

        Returns:
            An action (int).
        """

        # Define the actions
        if allowd_actions is not None:
            actions = allowd_actions
        else:
            actions = self.actions

        # With probabilty epsilon, pick random action
        random_digit = np.random.random_sample()  # Get random digit
        if (
            random_digit < self.eps
        ):  # If epsilon is smaller than the random digit, pick random action
            action = np.random.choice(actions)
        # If epsilon is bigger than the random digit, pick random greedy action.
        # Note that if there is only one greedy action this will always be selected
        else:  # If epsilon is bigger than the random digit, pick random greedy action.
            Q = self.load_weights(model_id)
            Q_state = Q[state]
            _new_q = [
                Q_state[i] if i in actions else 0.0 for i in range(len(self.actions))
            ]
            new_q = np.array(_new_q)
            max_list = np.argwhere(new_q == np.amax(new_q)).flatten()

            intersection = [x for x in actions if x in max_list]
            action = np.random.choice(intersection)

        return action

    def learn(self, state, action, reward, next_state, model_id):
        Q = self.load_weights(model_id)
        s_ = next_state
        s = state
        a = action
        r = reward
        # Compute policy probabilities
        policy_probs = [
            self.eps / self.actions_n
        ] * self.actions_n  # Every action at least epsilon/number_actions prob
        max_actions = (
            np.argwhere(self.Q[s_] == np.amax(self.Q[s_])).flatten().tolist()
        )  # Get all maximum actions
        for index in max_actions:
            policy_probs[index] += (1 - self.eps) / len(
                max_actions
            )  # Add minimum probability for max actions with (1-epsilon)/number of maxium actions

        Q[s][a] = Q[s][a] + self.alpha * (
            r + self.gamma * np.sum(Q[s_] * policy_probs) - Q[s][a]
        )  # Update step

        self.save_weights(model_id, Q)

    def get_weights(self, model_id):
        # return self.weights, self.biases
        Q = self.load_weights(model_id)
        return Q

    def set_weights(self, Q):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        self.Q = copy.deepcopy(Q)

    def model_params_key(self, model_id):
        return f"{model_id}:esarsa"

    def save_weights(self, model_id, Q):
        model_key = self.model_params_key(model_id)
        self._model_db.set(model_key, pickle.dumps(Q))

    def load_weights(self, model_id=None):
        try:
            model_key = self.model_params_key(model_id)
            model = self._model_db.get(model_key)
            if model is not None:
                Q = pickle.loads(model)
                return Q
            else:
                return self.Q
        except:
            print("Could not load weights: File Not Found, use default")
            print(traceback.format_exc())

            return self.Q
