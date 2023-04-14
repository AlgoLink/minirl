import abc
from typing import Optional
import numbers
import math
import copy
import numpy as np


from .base import Base


class Statistic(abc.ABC, Base):
    """A statistic."""

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abc.abstractmethod
    def get(self) -> Optional[float]:
        """Return the current value of the statistic."""

    def __repr__(self):
        try:
            value = self.get()
        except NotImplementedError:
            value = None
        fmt_value = None if value is None else f"{value:{self._fmt}}".rstrip("0")
        return f"{self.__class__.__name__}: {fmt_value}"

    def __str__(self):
        return repr(self)

    def __gt__(self, other):
        return self.get() > other.get()


class Univariate(Statistic):
    """A univariate statistic measures a property of a variable."""

    @abc.abstractmethod
    def update(self, x: numbers.Number):
        """Update and return the called instance."""
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def __or__(self, other):
        from .link import Link

        return Link(left=self, right=other)


class AbsMax(Univariate):
    """Running absolute max.
    Attributes
    ----------
    abs_max : float
        The current absolute max.
    Examples
    --------
    >>> from  minirl.preprocessing import stats
    >>> X = [1, -4, 3, -2, 5, -6]
    >>> abs_max = stats.AbsMax()
    >>> for x in X:
    ...     print(abs_max.update(x).get())
    1
    4
    4
    4
    5
    6
    """

    def __init__(self):
        self.abs_max = 0.0

    def update(self, x):
        if abs(x) > self.abs_max:
            self.abs_max = abs(x)
        return self

    def get(self):
        return self.abs_max


class Min(Univariate):
    """Running min.
    Attributes
    ----------
    min : float
        The current min.
    """

    def __init__(self):
        self.min = math.inf

    def update(self, x):
        if x < self.min:
            self.min = x
        return self

    def get(self):
        return self.min


class Max(Univariate):
    """Running max.
    Attributes
    ----------
    max : float
        The current max.
    Examples
    --------
    >>> from minirl.preprocessing import stats
    >>> X = [1, -4, 3, -2, 5, -6]
    >>> _max = stats.Max()
    >>> for x in X:
    ...     print(_max.update(x).get())
    1
    1
    3
    3
    5
    5
    """

    def __init__(self):
        self.max = -math.inf

    def update(self, x):
        if x > self.max:
            self.max = x
        return self

    def get(self):
        return self.max


class Mean(Univariate):
    """Running mean.
    Attributes
    ----------
    n : float
        The current sum of weights. If each passed weight was 1, then this is equal to the number
        of seen observations.
    Examples
    --------
    >>> from minirl.preprocessing import stats
    >>> X = [-5, -3, -1, 1, 3, 5]
    >>> mean = stats.Mean()
    >>> for x in X:
    ...     print(mean.update(x).get())
    -5.0
    -4.0
    -3.0
    -2.0
    -1.0
    0.0
    You can calculate a rolling average by wrapping a `utils.Rolling` around:
    >>> from minirl import utils
    >>> X = [1, 2, 3, 4, 5, 6]
    >>> rmean = utils.Rolling(stats.Mean(), window_size=2)
    >>> for x in X:
    ...     print(rmean.update(x).get())
    1.0
    1.5
    2.5
    3.5
    4.5
    5.5
    References
    ----------
    [^1]: [West, D. H. D. (1979). Updating mean and variance estimates: An improved method. Communications of the ACM, 22(9), 532-535.](https://dl.acm.org/doi/10.1145/359146.359153)
    [^2]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
    [^3]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0

    def update(self, x, w=1.0):
        self.n += w
        self._mean += (w / self.n) * (x - self._mean)
        return self

    def update_many(self, X: np.ndarray):
        a = self.n / (self.n + len(X))
        b = len(X) / (self.n + len(X))
        self._mean = a * self._mean + b * np.mean(X)
        self.n += len(X)
        return self

    def revert(self, x, w=1.0):
        self.n -= w
        if self.n < 0:
            raise ValueError("Cannot go below 0")
        elif self.n == 0:
            self._mean = 0.0
        else:
            self._mean -= (w / self.n) * (x - self._mean)
        return self

    def get(self):
        return self._mean

    @classmethod
    def _from_state(cls, n, mean):
        new = cls()
        new.n = n
        new._mean = mean

        return new

    def __iadd__(self, other):
        old_n = self.n
        self.n += other.n
        self._mean = (old_n * self._mean + other.n * other.get()) / self.n
        return self

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other
        return result

    def __isub__(self, other):
        old_n = self.n
        self.n -= other.n

        if self.n > 0:
            self._mean = (old_n * self._mean - other.n * other._mean) / self.n
        else:
            self.n = 0.0
            self._mean = 0.0
        return self

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other
        return result
