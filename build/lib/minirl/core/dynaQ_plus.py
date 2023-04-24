from copy import deepcopy
import numpy as np
from collections import deque
import pickle

# Time-based model for planning in Dyna-Q+
class TimeModel:
    # @actions: the actions instance. Indeed it's not very reasonable to give access to actions to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, actions, time_weight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.actions = actions

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1
        if state not in self.model.keys():
            self.model[state] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[state][action_] = [state, 0, 1]

        self.model[state][action] = [next_state, reward, self.time]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return state, action, next_state, reward


class DynaQPlus:
    def __init__(
        self,
        actions,
        eps=0.2,
        n=1,
        gamma=0.95,
        alpha=0.7,
        alpha_decay=0.001,
        model_db=None,
        score_db=None,
        his_db=None,
    ):
        self.actions = actions
        self.eps = eps
        self._model_db = model_db
        self._score_db = score_db
        self._his_db = his_db

        self.rewards = deque(maxlen=n)  # (up to) n last rewards seen
        self.stateActionHist = deque(maxlen=n)  # (up to) n last (s, a) seen
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        # Stabilize and converge to optimal policy
        self.alpha_decay = alpha_decay  # 600 episodes to fully decay
        self.q_model = {}

    def act(
        self, state, model_id=None, request_id=None, topN=1, eps=None, save_his=False
    ):
        _eps = eps if eps is not None else self.eps

        if np.random.binomial(1, _eps) == 1:
            action = self.get_random_action(topN)
        else:
            action = self.greedy_action_selection(state, model_id, topN)

        if save_his:
            self.update_state_action_hist(request_id, state, action, None, 0, model_id)

        return action

    def get_state_action_hist(
        self, model_id, log_type="action_pair", result_num="0", del_old_his=True
    ):
        if log_type == "reward":
            key = f"{model_id}:Reward"
        else:
            key = f"{model_id}:Hist"

        result = self._his_db.lrange(key, "0", result_num)
        list_len = self._his_db.llen(key)
        if del_old_his and list_len > self.n:
            self._his_db.delete(key)
        return result

    def update_state_action_hist(
        self,
        request_id=None,
        state=None,
        action=None,
        next_state=None,
        reward=None,
        model_id=None,
        log_type="action_pair",
    ):
        if log_type == "reward":
            key = f"{model_id}:Reward"
            value = pickle.dumps([request_id, reward])
        else:
            key = f"{model_id}:Hist"
            value = pickle.dumps([request_id, state, action, next_state])

        self._his_db.rlpush(key, value)
        return True

    def get_state_action_hist2(self, model_id, type="action_pair"):
        state_action_hist_key = f"{model_id}:Hist"
        state_action_hist = pickle.loads(
            self._his_db.get(state_action_hist_key)
            or pickle.dumps(self.stateActionHist)
        )
        return state_action_hist

    def update_state_action_hist2(self, state, action, model_id):
        state_action_hist = self.get_state_action_hist(model_id)
        state_action_hist.append([state, action])
        state_action_hist_key = f"{model_id}:Hist"
        self._his_db.set(state_action_hist_key, pickle.dumps(state_action_hist))
        return True

    def get_random_action(self, topN):
        if topN > len(self.actions):
            raise Exception("topN is longer than len of self.actions")
        return np.random.choice(self.actions, size=topN, replace=False, p=None).tolist()

    def greedy_action_selection(self, state, model_id=None, topN=1, withscores=False):
        """
        Selects action with the highest Q-value for the given state.
        """
        # Get all the Q-values for all possible actions for the state
        maxQ_action_list = self.get_maxQ(state, model_id, topN, withscores=withscores)
        if len(maxQ_action_list) < 1:
            maxQ_action_list = self.get_random_action(topN)
        return maxQ_action_list

    def get_maxQ(self, state, model_id, topN, withscores=False):
        score_key = f"{model_id}:{state}:Qscore"
        if withscores:
            score_list = self._score_db.zrange(
                score_key, "0", str(topN - 1), "withscores"
            )
        else:
            score_list = self._score_db.zrange(score_key, "0", str(topN - 1))
        return score_list

    def get_Q_state_value(self, state, action, model_id):
        # next state
        model_key = f"{model_id}:{state}:{action}:qvalue"
        # model_key = f"{local_model_id}:{state}:qvalue"
        _Q_stateValue = self._model_db.get(model_key)
        if _Q_stateValue is None:
            Q_stateValue = 0
        else:
            Q_stateValue = float(_Q_stateValue)

        return Q_stateValue

    def update_q_score(self, state, qvalue, action, model_id):
        score_key = f"{model_id}:{state}:Qscore"

        if float(qvalue) > 0:
            Q_score = "-{}".format(qvalue)
        else:
            Q0_state_value = abs(qvalue)
            Q_score = "{}".format(Q0_state_value)

        self._score_db.zadd(score_key, Q_score, str(action))

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

    def learn(
        self, request_id, reward, model_id=None, planning_steps=5, dyna_model=None
    ):
        # Q-Learning update
        recom_his = self.get_state_action_hist(model_id)
        q_model = self.get_q_model(model_id)
        self.q_model = q_model
        state = None
        action = None
        next_state = None
        if recom_his is not None:
            _request_id, _state, _action, _next_state = pickle.loads(recom_his)[0]
            if str(_request_id) == str(request_id):

                argmax_Q1_action = self.greedy_action_selection(_next_state, model_id)[
                    0
                ]

                # q_model = self.get_q_model(model_id)
                Q1_state_value = self.get_Q_value(
                    self.q_model, _next_state, argmax_Q1_action
                )
                Q0_state_value = self.get_Q_value(self.q_model, _state, _action)

                td_error = reward + self.gamma * Q1_state_value - Q0_state_value
                Q0_state_value += self.alpha * td_error

                self.q_model[(_state, _action)] = Q0_state_value
                # self.update_q_model(self.q_model, model_id)
                self.update_q_score(_state, Q0_state_value, _action, model_id)
                state = _state
                action = _action
                next_state = _next_state

        # feed the model with experience
        if dyna_model is not None and state is not None:
            dyna_model.feed(state, action, next_state, reward)

        # sample experience from the model
        if len(dyna_model.model.keys()) > planning_steps:
            # q_model = self.get_q_model(model_id)
            for t in range(0, planning_steps):
                state_, action_, next_state_, reward_ = dyna_model.model.sample()
                argmax_Q1_action = self.greedy_action_selection(next_state_, model_id)[
                    0
                ]
                Q1_state_value = self.get_Q_value(
                    self.q_model, next_state_, argmax_Q1_action
                )
                Q0_state_value = self.get_Q_value(self.q_model, state_, action_)

                td_error = reward_ + self.gamma * Q1_state_value - Q0_state_value
                Q0_state_value += self.alpha * td_error

                self.q_model[(state_, action_)] = Q0_state_value
                self.update_q_score(state_, Q0_state_value, action_)

        self.update_q_model(self.q_model, model_id)
        return True

    def update_q_model(self, Q_dict, model_id):
        model_key = f"{model_id}:qvalue"
        self._model_db.set(model_key, pickle.dumps(Q_dict))

        return model_key

    def get_q_model(self, model_id):
        model_key = f"{model_id}:qvalue"
        _model = self._model_db.get(model_key)
        if _model is None:
            model = {}
        else:
            model = pickle.loads(_model)
        return model
