import pickle
import numpy as np
from collections import deque
import random

DEFAULT_PREFIX = "dQ"


def _key(k):
    return "{0}:{1}".format(DEFAULT_PREFIX, k)


class DynaQ:
    def __init__(
        self,
        state_space,
        actions,
        alpha=0.7,
        gamma=0.9,
        random_seed=2023,
        eps=0.2,
        model_db=None,
        score_db=None,
        his_db=None,
        N=7,  # no. of steps in planning phase
        n=3,  # TD(n)
    ):
        self.gamma = gamma
        self.alpha = alpha
        self._model_db = model_db
        self._his_db = his_db
        self._score_db = score_db
        self._eps = eps
        self._N = N  # no. of steps in planning phase
        self.n = n  # no. of steps in return computation (n-step return)
        self.action_list = actions
        self.n_actions = len(actions)
        self.state_space = state_space
        self.rewards = deque(maxlen=n)  # (up to) n last rewards seen
        self.stateActionHist = deque(maxlen=n)  # (up to) n last (s, a) seen

        np.random.seed(random_seed)
        n_hidden = 64
        n_feature = 20
        self.model = dict()  # 环境模型
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

    def act(
        self,
        state,
        local_model_id=None,
        share_model_id=None,
        update_his=True,
        topN=1,
        eps=None,
        use_doubleQ=True,
        debug=False,
    ):
        # Choose a random action
        if eps is None:
            explore = np.random.binomial(1, self._eps)
        else:
            explore = np.random.binomial(1, eps)

        if explore == 1:
            if debug:
                print(f"act using random method")
            # action = random.choice(self.actions)
            action = self.get_random_action(topN)[0]
        # Choose the greedy action
        else:
            if use_doubleQ:
                if np.random.rand() <= 0.5:
                    doubleQ = 1
                else:
                    doubleQ = 2
            else:
                doubleQ = 1
            if debug:
                print(f"using Q {doubleQ}")
            _action = self.greedy_action_selection(state, local_model_id, doubleQ, topN)
            if len(_action) < 1:
                print("random")
                action = self.get_random_action(topN)[0]
            else:
                action = _action[0]
        # # stateActFreq[s, a] = how many times we've been in state s and taken action a
        if update_his:
            self.update_stateActionHist(state, action, local_model_id)
        if share_model_id is not None:
            self.update_stateActFreq(share_model_id, state, action)

        return action

    def q_learning(self, s0, a0, r, s1, model_id):
        argmax_Q1_action = self.greedy_action_selection(s1, model_id, doubleQ=1)[0]
        Q1_state_value = self.get_Q_state_value(
            s1, argmax_Q1_action, model_id, model_id_suffix="qvalue1"
        )
        Q0_state_value = self.get_Q_state_value(
            s0, a0, model_id, model_id_suffix="qvalue1"
        )
        td_error = r + self.gamma * Q1_state_value - Q0_state_value
        Q0_state_value += self.alpha * td_error

        model_key_start1 = f"{model_id}:{s0}:{a0}:qvalue1"
        self._model_db.set(model_key_start1, str(Q0_state_value))

        self.update_q_score(s0,Q0_state_value,a0,model_id,"q1")

    def update_q_score(self,state,qvalue,action, model_id,q_suffix="q1"):
        if q_suffix=="q1":
            score_key = f"{model_id}:{state}:Qscore1"
        else:
            score_key = f"{model_id}:{state}:Qscore2"
        if float(qvalue)>0:
            Q_score = "-{}".format(qvalue)
        else:
            Q0_state_value = abs(qvalue)
            Q_score = "{}".format(Q0_state_value)

        self._score_db.zadd(score_key, Q_score, str(action))
        

    def update(self, s0, a0, r, s1, model_id):
        self.q_learning(s0, a0, r, s1, model_id)
        self.model[(s0, a0)] = r, s1  # 将数据添加到模型中
        for _ in range(self._N):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_, model_id)

    def greedy_action_selection(self, state, model_id=None, doubleQ=1, topN=1):
        """
        Selects action with the highest Q-value for the given state.
        """
        # Get all the Q-values for all possible actions for the state
        maxQ_action_list = self.get_maxQ(
            state, model_id, topN, doubleQ, withscores=False
        )
        if len(maxQ_action_list) < 1:
            maxQ_action_list = self.get_random_action(topN)
        return maxQ_action_list

    def get_maxQ(self, state, model_id, topN, doubleQ=1, withscores=False):
        score_key1 = f"{model_id}:{state}:Qscore1"
        score_key2 = f"{model_id}:{state}:Qscore2"
        if doubleQ == 1:
            score_key = score_key1
        else:
            score_key = score_key2

        if withscores:
            score_list = self._score_db.zrange(
                score_key, "0", str(topN - 1), "withscores"
            )
        else:
            score_list = self._score_db.zrange(score_key, "0", str(topN - 1))
        return score_list

    def get_random_action(self, topN):
        if topN > len(self.action_list):
            raise Exception("topN is longer than len of self.actions")
        action_list = np.random.choice(
            self.action_list, size=topN, replace=False, p=None
        ).tolist()
        return action_list

    def learn(
        self,
        state,
        reward,
        local_model_id=None,
        share_model_id=None,
        use_doubleQ=True,
        use_dyna=True,
        eps=0.2,
        debug=False,
    ):
        rewards = self.update_rewards(reward, local_model_id)
        # determine whether Q should be updated yet, depending on whether we have observed n rewards yet (for n-step return)
        updateQ = len(rewards) >= self.n
        # update Q(s, a) with data (s', r)
        if updateQ:
            cumRew = self.computeMultiStepReturn(rewards)
            originalStateAction = self.get_stateActionHist(local_model_id)
            startState = originalStateAction[0][0]
            startAction = originalStateAction[0][1]
            # update either Q1 or Q2 with 50 % chance
            if use_doubleQ:
                if np.random.rand() <= 0.5:
                    argmax_Q1_action = self.greedy_action_selection(
                        state, local_model_id, doubleQ=1
                    )[0]
                    Q2_state_value = self.get_Q_state_value(
                        state, argmax_Q1_action, local_model_id, "qvalue2"
                    )
                    Q1_start_state_value = self.get_Q_state_value(
                        startState,
                        startAction,
                        local_model_id,
                        model_id_suffix="qvalue1",
                    )

                    delta = (
                        cumRew
                        + self.gamma ** len(rewards) * Q2_state_value
                        - Q1_start_state_value
                    )

                    Q1_start_state_value += self.alpha * delta
                    model_key_1 = f"{local_model_id}:{startState}:{startAction}:qvalue1"
                    self._model_db.set(model_key_1, str(Q1_start_state_value))

                    score_key = f"{local_model_id}:{startState}:Qscore1"
                    action = startAction
                    qvalue = Q1_start_state_value
                    self.update_q_score(startState,qvalue,action,local_model_id,"q1")

                else:
                    argmax_Q2_action = self.greedy_action_selection(
                        state, local_model_id, doubleQ=2
                    )[0]

                    Q1_state_value = self.get_Q_state_value(
                        state,
                        argmax_Q2_action,
                        local_model_id,
                        model_id_suffix="qvalue1",
                    )
                    Q2_start_state_action_value = self.get_Q_state_value(
                        startState,
                        startAction,
                        local_model_id,
                        model_id_suffix="qvalue2",
                    )

                    delta = (
                        cumRew
                        + self.gamma ** len(rewards) * Q1_state_value
                        - Q2_start_state_action_value
                    )

                    Q2_start_state_action_value += self.alpha * delta

                    model_key_2 = f"{local_model_id}:{startState}:{startAction}:qvalue2"
                    self._model_db.set(model_key_2, str(Q2_start_state_action_value))
                    score_key = f"{local_model_id}:{startState}:Qscore2"
                    action = startAction
                    qvalue = Q2_start_state_action_value
                    self.update_q_score(startState,qvalue,action,local_model_id,"q2")
            else:
                argmax_Q1_action = self.greedy_action_selection(
                    state, local_model_id, doubleQ=1
                )[0]
                Q1_state_value = self.get_Q_state_value(
                    state, argmax_Q1_action, local_model_id, model_id_suffix="qvalue1"
                )
                Q1_start_state_action_value = self.get_Q_state_value(
                    startState, startAction, local_model_id, model_id_suffix="qvalue1"
                )

                delta = (
                    cumRew
                    + self.gamma ** len(rewards) * Q1_state_value
                    - Q1_start_state_action_value
                )

                Q1_start_state_action_value += self.alpha * delta

                model_key_start1 = (
                    f"{local_model_id}:{startState}:{startAction}:qvalue1"
                )
                self._model_db.set(model_key_start1, str(Q1_start_state_action_value))

                score_key = f"{local_model_id}:{startState}:Qscore1"
                action = startAction
                qvalue = Q1_start_state_action_value
                self.update_q_score(startState,qvalue,action,local_model_id,"q1")
            
            if share_model_id is not None:
                self.update_lastDelta(startState, startAction, delta, share_model_id)
        # update Model(s, a) with (s', r)
        if share_model_id is not None:
            stateActionHist = self.get_stateActionHist(local_model_id)
            if len(stateActionHist) > 0:
                lastState = stateActionHist[-1][0]
                lastAction = stateActionHist[-1][1]
                self.updateStateStats(lastState, lastAction, state, share_model_id)

                self.update_avgRew(lastState, lastAction, float(reward), share_model_id)

        if use_dyna:
            # planning phase: update Q1/Q2 N times (1-step error instead of n-step)
            if share_model_id is not None:
                for _ in range(self._N):
                    s = np.random.choice(self.state_space)
                    explore = np.random.binomial(1, eps)
                    if explore == 1:
                        a = np.random.choice(self.action_list)
                    else:
                        a = self.get_action_bylastDelta(s, share_model_id, topN=1)
                    # query model and update Q1 or Q2 with observation data (s', r)
                    nextState, reward = self.queryShareModel(
                        s, a, next_state=None, model_id=share_model_id
                    )

                    if use_doubleQ:
                        if np.random.rand() <= 0.5:
                            argmax_Q1_action = self.greedy_action_selection(
                                nextState, local_model_id, doubleQ=1
                            )[0]
                            # next state
                            Q2_state_value = self.get_Q_state_value(
                                nextState, argmax_Q1_action, local_model_id, "qvalue2"
                            )
                            Q1_start_state_value = self.get_Q_state_value(
                                s,
                                a,
                                local_model_id,
                                model_id_suffix="qvalue1",
                            )

                            delta = (
                                reward
                                + self.gamma * Q2_state_value
                                - Q1_start_state_value
                            )

                            Q1_start_state_value += self.alpha * delta

                            model_key1 = f"{local_model_id}:{s}:{a}:qvalue1"
                            self._model_db.set(model_key1, str(Q1_start_state_value))
                            score_key = f"{local_model_id}:{s}:Qscore1"
                            action = a
                            qvalue = Q1_start_state_value
                            self.update_q_score(s,qvalue,action,local_model_id,"q1")
                        else:
                            argmax_Q2_action = self.greedy_action_selection(
                                nextState, local_model_id, doubleQ=2
                            )[0]
                            # next state
                            Q1_state_value = self.get_Q_state_value(
                                nextState, argmax_Q2_action, local_model_id, "qvalue1"
                            )
                            Q2_start_state_value = self.get_Q_state_value(
                                s,
                                a,
                                local_model_id,
                                model_id_suffix="qvalue2",
                            )

                            delta = (
                                reward
                                + self.gamma * Q1_state_value
                                - Q2_start_state_value
                            )

                            Q2_start_state_value += self.alpha * delta

                            model_key2 = f"{local_model_id}:{s}:{a}:qvalue2"
                            self._model_db.set(model_key2, str(Q2_start_state_value))
                            score_key = f"{local_model_id}:{s}:Qscore2"
                            action = a
                            qvalue = Q2_start_state_value
                            self.update_q_score(s,qvalue,action,local_model_id,"q2")
                    else:
                        argmax_Q1_action = self.greedy_action_selection(
                            nextState, local_model_id, doubleQ=1
                        )[0]

                        Q1_state_value = self.get_Q_state_value(
                            nextState, argmax_Q1_action, local_model_id, "qvalue1"
                        )

                        Q1_start_state_value = self.get_Q_state_value(
                            s, a, local_model_id, "qvalue1"
                        )

                        model_key1 = (
                            f"{local_model_id}:{nextState}:{argmax_Q1_action}:qvalue1"
                        )

                        model_key_start1 = f"{local_model_id}:{s}:{a}:qvalue1"
                        Q1_start_state_value += self.alpha * delta

                        self._model_db.set(model_key_start1, str(Q1_start_state_value))
                        score_key = f"{local_model_id}:{s}:Qscore1"
                        action = a
                        qvalue = Q1_start_state_value
                        self.update_q_score(s,qvalue,action,local_model_id,"q1")

                    self.update_lastDelta(s, a, delta, share_model_id)

        return 0

    def get_Q_state_value(self, state, action, model_id, model_id_suffix="qvalue1"):
        # next state
        model_key = f"{model_id}:{state}:{action}:{model_id_suffix}"
        # model_key = f"{local_model_id}:{state}:qvalue"
        _Q_stateValue = self._model_db.get(model_key)
        if _Q_stateValue is None:
            Q2_stateValue = 0
        else:
            Q2_stateValue = float(_Q_stateValue)

        return Q2_stateValue

    def update_lastDelta(self, state, action, delta, model_id):
        delta_key = f"{model_id}:{state}:lastDelta"
        if float(delta)>0:
            delta_score = "-{}".format(delta)
        else:
            _delta = abs(delta)
            delta_score = "{}".format(_delta)

        self._score_db.zadd(delta_key, delta_score, str(action))
        return delta

    def get_action_bylastDelta(self, state, model_id, topN=1):
        delta_key = f"{model_id}:{state}:lastDelta"
        action_list = self._score_db.zrange(delta_key, "0", str(topN - 1))
        if len(action_list) < 1:
            action_list = np.random.choice(self.action_list, size=topN).tolist()
        return action_list[0]

    def computeMultiStepReturn(self, rewards):
        ret = 0
        for i, reward in enumerate(rewards):
            ret += reward * self.gamma**i
        return ret

    def get_stateActionHist(self, model_id):
        stateActionHist_key = f"{model_id}:Hist"
        _stateActionHist = self._his_db.get(stateActionHist_key)
        if _stateActionHist is None:
            stateActionHist = self.stateActionHist
        else:
            stateActionHist = pickle.loads(_stateActionHist)

        return stateActionHist

    def update_stateActionHist(self, state, action, model_id):
        stateActionHist = self.get_stateActionHist(model_id)
        stateActionHist.append([state, action])
        stateActionHist_key = f"{model_id}:Hist"
        self._his_db.set(stateActionHist_key, pickle.dumps(stateActionHist))

        return True

    def get_rewards(self, model_id):
        rewards_key = f"{model_id}:Rewards"
        _rewards = self._his_db.get(rewards_key)
        if _rewards is None:
            rewards = self.rewards
        else:
            rewards = pickle.loads(_rewards)

        return rewards

    def update_rewards(self, reward, model_id):
        rewards_key = f"{model_id}:Rewards"
        rewards = self.get_rewards(model_id)
        rewards.append(reward)
        self._his_db.set(rewards_key, pickle.dumps(rewards))
        return rewards

    def queryShareModel(
        self,
        state,
        action,
        next_state=None,
        model_id=None,
        topN=1,
        next_state_score="ns",
    ):
        # given (state, action), return nextState according to its relative frequency after (state, action) in the history
        # if (state, action) has never been seen before, return a random next state
        if next_state is None:
            best_next_state = self.getStateStats(
                model_id, next_state_score=next_state_score, topN=topN
            )
        else:
            best_next_state = next_state

        avg_reward_key = f"{model_id}:{state}:{action}:avgRew"

        _reward = self._model_db.get(avg_reward_key)
        if _reward is None:
            reward = 0
        else:
            reward = float(_reward)

        return best_next_state, reward

    def updateShareModel(self, state, action, model_id=None):
        avg_reward_key = f"{model_id}:{state}:{action}avgRew"
        stateActFreq_key = f"{model_id}:{state}:{action}:stateActFreq"
        state_action_expose = self._model_db.get(stateActFreq_key)

        return True

    def getStateStats(self, model_id, next_state_score="ns", topN=1):
        next_state_score_key = _key(f"{model_id}:{next_state_score}")
        best_next_state = self._score_db.zrange(
            next_state_score_key, "0", str(topN - 1)
        )[0]

        return best_next_state

    def updateStateStats(
        self, state, action, next_state, model_id=None, next_state_score="ns"
    ):
        """
            stateTransFreq[s, a, s'] = frequency of transition (s, a) -> s'
        params:
            state (str): user current state
            action (int): an integer for the action
            next_state(str): next state by action on current state
        """
        state_action_tries = self._increment_item_tries(f"{state}_{action}", model_id)
        state_action_next_tries = self._increment_item_tries(
            f"{state}_{action}_{next_state}", model_id
        )

        next_state_score_key = _key(f"{model_id}:{next_state_score}")
        _item_score = state_action_next_tries / (state_action_tries + 1e-7)

        model_score = "-{}".format(_item_score)
        self._score_db.zadd(next_state_score_key, model_score, str(next_state))
        return model_score

    # update cnt
    def _increment_item_tries(self, item: str, model_id: str) -> None:
        key_tries = _key(f"{model_id}:{item}:tries")
        item_tries = self._model_db.incr(
            key_tries
        )  # self.rlite_client.command("incr",key_tries)
        return item_tries

    # update custom itemkey cnt
    def _increment_customitem_tries(self, key: str) -> None:
        customitem_tries = self._model_db.incr(
            key
        )  # self.rlite_client.command("incr",key_tries)
        return customitem_tries

    def update_stateActFreq(self, model_id, state, action):
        stateActFreq_key = f"{model_id}:{state}:{action}:stateActFreq"
        return self._increment_customitem_tries(stateActFreq_key)

    def update_avgRew(self, state, action, reward, model_id=None):
        # update Model(s, a) with (s', r)
        # update stateActFreq when act
        # update update_avgRew when learn

        stateActFreq_key = f"{model_id}:{state}:{action}:stateActFreq"
        _stateActFreq = self._model_db.get(stateActFreq_key)
        if _stateActFreq is None:
            stateActFreq = self.update_stateActFreq(model_id, state, action)
        else:
            stateActFreq = int(_stateActFreq)

        avg_reward_key = f"{model_id}:{state}:{action}:avgRew"
        _old_avg_reward = self._model_db.get(avg_reward_key)
        if _old_avg_reward is None:
            old_avg_reward = 0
        else:
            old_avg_reward = float(_old_avg_reward)
        new_avg_reward = old_avg_reward + (reward - old_avg_reward) / stateActFreq

        self._model_db.set(avg_reward_key, str(new_avg_reward))
        return new_avg_reward
