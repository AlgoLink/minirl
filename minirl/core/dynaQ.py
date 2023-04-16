import pickle
import numpy as np


class DynaQ:
    def __init__(
        self,
        actions,
        alpha=0.7,
        gamma=0.9,
        random_seed=2023,
        eps=0.2,
        model_db=None,
        score_db=None,
        his_db=None,
        N=7,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self._model_db = model_db
        self._his_db = his_db
        self._score_db = score_db
        self._eps = eps
        self._N = N

        np.random.seed(random_seed)
        n_hidden = 64
        n_feature = 20
        print("simple two layer neural network based on numpy")
        print(f"creating nn: #input:{n_feature} #hidden:{n_hidden} #output:{actions}")

    def get_Q_value(self, Q_dict, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        return Q_dict.get(
            (state, action), 0.0
        )  # Return 0.0 if state-action pair does not exist

    def act(self, state, epsilon=0.1, model_id=None, topN=1, eps=None):
        # Choose a random action
        if eps is None:
            explore = np.random.binomial(1, self._eps)
        else:
            explore = np.random.binomial(1, eps)

        if explore == 1:
            # action = random.choice(self.actions)
            action = self.get_random_action(topN)
        # Choose the greedy action
        else:
            if np.random.rand() <= 0.5:
                q = self.Q1[observation, :]
            else:
                q = self.Q2[observation, :]

            action = self.greedy_action_selection(state, model_id, topN)
            if len(action) < 1:
                action = self.get_random_action(topN)

        return action

    def learn(
        self, state, action, reward, next_state, model_id=None, reward_type="avg"
    ):

        return True

    def queryModel(self, state, action, model_id=None):
        # given (state, action), return nextState according to its relative frequency after (state, action) in the history
        # if (state, action) has never been seen before, return a random next state

        stateTransFreq_key = f"{model_id}:transFreq"
        _stateTransFreq = self._model_db.get(stateTransFreq_key)
        if _stateTransFreq is None:
            stateTransFreq = 0
        else:
            stateTransFreq = pickle.loads(_stateTransFreq)
        if stateTransFreq == 0:
            return None
        else:
            nextState = np.random.choice(
                self.state_space,
                p=self.stateTransFreq[state, action, :]
                / sum(self.stateTransFreq[state, action, :]),
            )
        reward = self.avgRew[state, action]
        return nextState, reward
