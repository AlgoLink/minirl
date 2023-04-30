""" Package for numpy array implementations """
from __future__ import absolute_import

import numpy
from minirl.neural_nets.array_variants.numpy import numpy_core

array_type = numpy.ndarray
number_type = [numpy.float16, numpy.float32, numpy.float64, numpy.int32, numpy.int64]

register_primitives = numpy_core.register_primitives
def_grads = numpy_core.def_grads
