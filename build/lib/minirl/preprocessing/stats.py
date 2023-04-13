import abc
from typing import Optional
import numbers
import math

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
