"""Wrapper for NumPy random functions."""
from __future__ import absolute_import
from __future__ import print_function

from .numpy_wrapper import wrap_namespace

import numpy


def register_primitives(reg, prim_wrapper):
    """Register primitives"""
    wrap_namespace(numpy.random.__dict__, reg, prim_wrapper)


def def_grads(prims):
    """Define gradients of primitives"""
    prims("random").def_grad_zero()
    prims("randn").def_grad_zero()
