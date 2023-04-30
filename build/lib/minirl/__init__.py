"""minirl nets root module"""
from __future__ import absolute_import

from .core.pg import PGAgent
from .core.ac import ACAgent
from .preprocessing.scaler import StandardScaler as StdScaler
from .preprocessing.scaler import MinMaxScaler
from .preprocessing.scaler import MaxAbsScaler
from .core.bandit import BanditEEAgent
from .core.dynaQ_plus import DynaQPlus, simpleModel, TimeModel
from .core.bandit_policy import LinUCB
from .core.bandit import ContextualLinearBandit as CBandit
from .core.one_step_pg import uniAgent


import re

from minirl.neural_nets.dispatch import policy

# from .dispatch.policy import PreferMXNetPolicy
from minirl.neural_nets.dispatch.policy import OnlyNumPyPolicy

# Global config
Config = {
    "modules": [],
}
# Config['default_policy'] = PreferMXNetPolicy()
Config["default_policy"] = OnlyNumPyPolicy()

# Import minpy.numpy package to do some initialization.
from minirl.neural_nets import numpy  # pylint: disable= wrong-import-position


def set_global_policy(plc):
    """Set global policy for all modules. This will also change default policy
    in future imported modules.

    Parameters
    ----------
    plc : str or Policy object
        The policy to set.
    """
    Config["default_policy"] = policy.create(plc)
    for mod in Config["modules"]:
        mod.generate_attrs(Config["default_policy"])


def get_global_policy():
    """Return the current global policy."""
    return Config["default_policy"]


wrap_policy = policy.wrap_policy
