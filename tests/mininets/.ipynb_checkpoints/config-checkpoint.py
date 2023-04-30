import numpy as np
import minirl.neural_nets.nn.init
import minirl.neural_nets.nn.optim
from minirl.neural_nets.nn.init import custom
from minirl.neural_nets.nn.optim import rmsprop

class Config(object):
    def __init__(self, args):
        # Default training settings
        self.init_func = custom
        self.init_config = {
            'function': lambda shape: np.random.randn(shape[0], shape[1]) / np.sqrt(shape[1])
        }
        self.learning_rate = 1e-3
        self.update_rule = rmsprop
        self.grad_clip = True
        self.clip_magnitude = 40.0

        # Default model settings
        self.hidden_size = 200
        self.gamma = 0.99
        self.lambda_ = 1.0
        self.vf_wt = 0.5        # Weight of value function term in the loss
        self.entropy_wt = 0.01  # Weight of entropy term in the loss

        # Override defaults with values from `args`.
        for arg in args:
            if arg in args:
                self.__setattr__(arg, args[arg])
