class BaseAgent:
    def __init__(self, model_db=None):
        self._model_db = model_db

    def act(self):
        """
        Select the action.
        """
        pass

    def learn(self):
        """
        Update the stochastic rl model.
        """
        pass
