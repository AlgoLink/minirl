""" Package for different implementations of array computations """
from __future__ import absolute_import
from __future__ import print_function

from .numpy.numpy_core import array_type, number_type
from ..utils import common


class ArrayType(object):
    """Enumeration of types of arrays."""

    # pylint: disable=no-init
    NUMPY = 0
    MXNET = 1


variants = {"numpy": ArrayType.NUMPY}
array_types = {"numpy": array_type}
variants_repr = {ArrayType.NUMPY: "NumPy"}
number_types = {"native": [int, float], "numpy": number_type}
