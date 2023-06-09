import numpy as np
import copy  # for deepcopy of model parameters
import pickle
import traceback


def argmax_rand(arr):
    # break ties randomly, np.argmax() always picks first max
    return np.random.choice(np.flatnonzero(arr == arr.max()))


def softmax(x):
    """Numerically stable softmax"""
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


class BanditAgent:
    def __init__(self, config):
        self.config = config
        self._init_model(config=config)

    def _init_model(self, config):
        self.algo_type = config.get("algo_type", "egreedy")
        self.action_n = config.get("action_n", 3)
        self.eps = config.get("eps", 0.2)
        self.c = config.get("c", 0.2)  # ucb
        self.alpha = config.get("alpha", 0.1)  # Q[A] += alpha * (R - Q[A]) opt
        self._model_db = config.get("model_db", None)
        self.Q = np.zeros(self.action_n)
        self.N = np.zeros(self.action_n)
        self.N_ALL = 0
        self.H = np.zeros(self.action_n)  # gradient

    @property
    def params(self):
        """A dictionary containing the current policy parameters"""
        return {"Q": self.Q, "N": self.N}

    @property
    def hyperparams(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": self.algo_type,
            "allowed_algo_type": ["greedy_opt", "egreedy", "gradient", "ucb"],
            "eps": self.eps,
            "alpha": self.alpha,
            "action_n": self.action_n,
            "model_db": self._model_db,
            "config": self.config,
        }

    def act(self, model_id):
        model = self.load_weights(model_id)
        self.save_weights(model)

        if self.algo_type == "egreedy":
            action = (
                argmax_rand(self.Q)
                if np.random.rand() > self.eps
                else np.random.randint(self.action_n)
            )
            self.N[action] += 1
            self.save_weights(model_id, [self.Q, self.N])

        elif self.algo_type == "greedy_opt":
            action = (
                argmax_rand(self.Q)
                if np.random.rand() > self.eps
                else np.random.randint(self.action_n)
            )

        elif self.algo_type == "gradient":
            pi = softmax(self.H)
            action = np.random.choice(range(self.action_n), p=pi)
        else:
            action = argmax_rand(self.Q + self.c * np.sqrt(np.log(self.N_ALL) / self.N))
            self.N[action] += 1
            self.N_ALL += 1
            self.save_weights(model_id, [self.Q, self.N, self.N_ALL])
        self._init_model()
        return action

    def learn(self, action, reward, model_id=None):
        model = self.load_weights(model_id=model_id)
        self.set_weights(model)
        A = action
        R = reward
        if self.algo_type == "egreedy":
            self.Q[A] += (1 / self.N[A]) * (R - self.Q[A])
            model = [self.Q, self.N]

        elif self.algo_type == "greedy_opt":
            self.Q[A] += self.alpha * (R - self.Q[A])
            model = self.Q

        elif self.algo_type == "gradient":
            pass
        else:
            self.Q[A] += (1 / self.N[A]) * (R - self.Q[A])
            model = [self.Q, self.N, self.N_ALL]

        self.save_weights(model_id=model_id, model=model)
        self._init_model(self.config)

    def get_weights(self, model_id):
        # return self.weights, self.biases
        model = self.load_weights(model_id)
        return model

    def set_weights(self, model):
        # use deepcopy to avoid target_model and normal model from using
        # the same weights. (standard copy means object references instead of
        # values are copied)
        model = copy.deepcopy(model)
        if self.algo_type == "egreedy":
            self.Q, self.N = model
        elif self.algo_type == "greedy_opt":
            self.Q = model
        elif self.algo_type == "gradient":
            pass
        else:
            self.Q, self.N, self.N_ALL = model

    def model_params_key(self, model_id):
        params_id = self.hyperparams.get("id")
        return f"{model_id}:{params_id}"

    def save_weights(self, model_id, model):
        if self._model_db is None:
            pickle.dump(
                model,
                open("{}.pickle".format(model_id), "wb"),
            )
        else:
            model_key = self.model_params_key(model_id)
            self._model_db.set(model_key, pickle.dumps(model))

    def load_weights(self, model_id=None):
        try:
            if self._model_db is None:
                Q = pickle.load(open("{}.pickle".format(model_id), "rb"))
            else:
                model_key = self.model_params_key(model_id)
                _model = self._model_db.get(model_key)
                if _model is not None:
                    model = pickle.loads(_model)
                    return model
                else:
                    if self.algo_type == "egreedy":
                        return [self.Q, self.N]
                    elif self.algo_type == "greedy_opt":
                        return self.Q
                    elif self.algo_type == "gradient":
                        pass
                    else:
                        return [self.Q, self.N, self.N_ALL]

        except:
            print("Could not load weights: File Not Found, use default")
            print(traceback.format_exc())

            if self.algo_type == "egreedy":
                return [self.Q, self.N]
            elif self.algo_type == "greedy_opt":
                return self.Q
            elif self.algo_type == "gradient":
                pass
            else:
                return [self.Q, self.N, self.N_ALL]
