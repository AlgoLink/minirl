import abc
import typing
import inspect
import collections
import copy
import sys
import itertools
import pandas as pd

from .utils import _repr_obj


class Base:
    """Base class that is inherited by the majority of classes in River.
    This base class allows us to handle the following tasks in a uniform manner:
    - Getting and setting parameters.
    - Displaying information.
    - Cloning.
    """

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return _repr_obj(obj=self)

    @classmethod
    def _unit_test_params(cls):
        """Instantiates an object with default arguments.
        Most parameters of each object have a default value. However, this isn't always the case,
        in particular for meta-models where the wrapped model is typically not given a default
        value. It's useful to have a default value set for testing reasons, which is the purpose of
        this method. By default it simply calls the __init__ function. It may be overridden on an
        individual as needed.
        """
        yield {}

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """Return the parameters that were used during initialization."""

        params = {}

        for name, param in inspect.signature(self.__init__).parameters.items():  # type: ignore
            # *args
            if param.kind == param.VAR_POSITIONAL:
                if positional_args := getattr(self, name, []):
                    params["_POSITIONAL_ARGS"] = positional_args
                continue

            # **kwargs
            if param.kind == param.VAR_KEYWORD:
                for k, v in getattr(self, name, {}).items():
                    if isinstance(v, Base):
                        params[k] = (v.__class__, v._get_params())
                    else:
                        params[k] = v
                continue

            # Keywords parameters
            attr = getattr(self, name)
            if isinstance(attr, Base):
                params[name] = (attr.__class__, attr._get_params())
            else:
                params[name] = attr

        return params

    def clone(self, new_params: dict = None, include_attributes=False):
        """Return a fresh estimator with the same parameters.
        The clone has the same parameters but has not been updated with any data.
        This works by looking at the parameters from the class signature. Each parameter is either
        - recursively cloned if its a class.
        - deep-copied via `copy.deepcopy` if not.
        If the calling object is stochastic (i.e. it accepts a seed parameter) and has not been
        seeded, then the clone will not be idempotent. Indeed, this method's purpose if simply to
        return a new instance with the same input parameters.
        Parameters
        ----------
        new_params
        include_attributes
            Whether attributes that are not present in the class' signature should also be cloned
            or not.
        Examples
        --------
        >>> from minirl import linear_model
        >>> from minirl import optim
        >>> model = linear_model.LinearRegression(
        ...     optimizer=optim.SGD(lr=0.042),
        ... )
        >>> new_params = {
        ...     'optimizer': optim.SGD(.001)
        ... }
        >>> model.clone(new_params)
        LinearRegression (
          optimizer=SGD (
            lr=Constant (
              learning_rate=0.001
            )
          )
          loss=Squared ()
          l2=0.
          l1=0.
          intercept_init=0.
          intercept_lr=Constant (
            learning_rate=0.01
          )
          clip_gradient=1e+12
          initializer=Zeros ()
        )
        The algorithm is recursively called down `Pipeline`s and `TransformerUnion`s.
        >>> from minirl import compose
        >>> from minirl import preprocessing
        >>> model = compose.Pipeline(
        ...     preprocessing.StandardScaler(),
        ...     linear_model.LinearRegression(
        ...         optimizer=optim.SGD(0.042),
        ...     )
        ... )
        >>> new_params = {
        ...     'LinearRegression': {
        ...         'optimizer': optim.SGD(0.03)
        ...     }
        ... }
        >>> model.clone(new_params)
        Pipeline (
          StandardScaler (
            with_std=True
          ),
          LinearRegression (
            optimizer=SGD (
              lr=Constant (
                learning_rate=0.03
              )
            )
            loss=Squared ()
            l2=0.
            l1=0.
            intercept_init=0.
            intercept_lr=Constant (
              learning_rate=0.01
            )
            clip_gradient=1e+12
            initializer=Zeros ()
          )
        )
        """

        def is_class_param(param):
            # See expand_param_grid to understand why this is necessary
            return (
                isinstance(param, tuple)
                and inspect.isclass(param[0])
                and isinstance(param[1], dict)
            )

        # Override the default parameters with the new ones
        params = self._get_params()
        params.update(new_params or {})

        # Clone by recursing
        clone = self.__class__(
            *(params.get("_POSITIONAL_ARGS", [])),
            **{
                name: (
                    getattr(self, name).clone(param[1])
                    if is_class_param(param)
                    else copy.deepcopy(param)
                )
                for name, param in params.items()
                if name != "_POSITIONAL_ARGS"
            },
        )

        if not include_attributes:
            return clone

        for attr, value in self.__dict__.items():
            if attr not in params:
                setattr(clone, attr, copy.deepcopy(value))
        return clone

    @property
    def _mutable_attributes(self) -> set:
        return set()

    def mutate(self, new_attrs: dict):
        """Modify attributes.
        This changes parameters inplace. Although you can change attributes yourself, this is the
        recommended way to proceed. By default, all attributes are immutable, meaning they
        shouldn't be mutated. Calling `mutate` on an immutable attribute raises a `ValueError`.
        Mutable attributes are specified via the `_mutable_attributes` property, and are thus
        specified on a per-estimator basis.
        Parameters
        ----------
        new_attrs
        Examples
        --------
        >>> from minirl import linear_model
        >>> from minirl import optim
        >>> model = linear_model.LinearRegression(
        ...     optimizer=optim.SGD(0.042),
        ... )
        >>> new_params = {
        ...     'optimizer': {'lr': optim.schedulers.Constant(0.001)}
        ... }
        >>> model.mutate(new_params)
        >>> model
        LinearRegression (
          optimizer=SGD (
            lr=Constant (
              learning_rate=0.001
            )
          )
          loss=Squared ()
          l2=0.
          l1=0.
          intercept_init=0.
          intercept_lr=Constant (
            learning_rate=0.01
          )
          clip_gradient=1e+12
          initializer=Zeros ()
        )
        The algorithm is recursively called down `Pipeline`s and `TransformerUnion`s.
        >>> from minirl import compose
        >>> from minirl import preprocessing
        >>> model = compose.Pipeline(
        ...     preprocessing.StandardScaler(),
        ...     linear_model.LinearRegression(
        ...         optimizer=optim.SGD(lr=0.042),
        ...     )
        ... )
        >>> new_params = {
        ...     'LinearRegression': {
        ...         'l2': 5,
        ...         'optimizer': {'lr': optim.schedulers.Constant(0.03)}
        ...     }
        ... }
        >>> model.mutate(new_params)
        >>> model
        Pipeline (
          StandardScaler (
            with_std=True
          ),
          LinearRegression (
            optimizer=SGD (
              lr=Constant (
                learning_rate=0.03
              )
            )
            loss=Squared ()
            l2=5
            l1=0.
            intercept_init=0.
            intercept_lr=Constant (
              learning_rate=0.01
            )
            clip_gradient=1e+12
            initializer=Zeros ()
          )
        )
        """

        def _mutate(obj, new_attrs):
            def is_class_attr(name, attr):
                return hasattr(getattr(obj, name), "mutate") and isinstance(attr, dict)

            for name, attr in new_attrs.items():
                if not hasattr(obj, name):
                    raise ValueError(
                        f"'{name}' is not an attribute of {obj.__class__.__name__}"
                    )

                # Check the attribute is mutable
                if name not in obj._mutable_attributes:
                    raise ValueError(
                        f"'{name}' is not a mutable attribute of {obj.__class__.__name__}"
                    )

                if is_class_attr(name, attr):
                    _mutate(obj=getattr(obj, name), new_attrs=attr)
                else:
                    setattr(obj, name, attr)

        _mutate(obj=self, new_attrs=new_attrs)

    @property
    def _is_stochastic(self):
        """Indicates if the model contains an unset seed parameter.
        The convention in River is to control randomness by exposing a seed parameter. This seed
        typically defaults to `None`. If the seed is set to `None`, then the model is expected to
        produce non-reproducible results. In other words it is not deterministic and is instead
        stochastic. This method checks if this is the case by looking for a None `seed` in the
        model's parameters.
        """

        def is_class_param(param):
            return (
                isinstance(param, tuple)
                and inspect.isclass(param[0])
                and isinstance(param[1], dict)
            )

        def find(params):
            if not isinstance(params, dict):
                return False
            for name, param in params.items():
                if name == "seed" and param is None:
                    return True
                if is_class_param(param) and find(param[1]):
                    return True
            return False

        return find(self._get_params())

    @property
    def _raw_memory_usage(self) -> int:
        """Return the memory usage in bytes."""

        import numpy as np

        buffer = collections.deque([self])
        seen = set()
        size = 0
        while len(buffer) > 0:
            obj = buffer.popleft()
            obj_id = id(obj)
            if obj_id in seen:
                continue
            size += sys.getsizeof(obj)
            # Important mark as seen to gracefully handle self-referential objects
            seen.add(obj_id)
            if isinstance(obj, dict):
                buffer.extend([k for k in obj.keys()])
                buffer.extend([v for v in obj.values()])
            elif hasattr(obj, "__dict__"):  # Save object contents
                contents: dict = vars(obj)
                size += sys.getsizeof(contents)
                buffer.extend([k for k in contents.keys()])
                buffer.extend([v for v in contents.values()])
            elif isinstance(obj, np.ndarray):
                size += obj.nbytes
            elif isinstance(obj, (itertools.count, itertools.cycle, itertools.repeat)):
                ...
            elif hasattr(obj, "__iter__") and not isinstance(
                obj, (str, bytes, bytearray)
            ):
                buffer.extend([i for i in obj])  # type: ignore

        return size

    @property
    def _memory_usage(self) -> str:
        """Return the memory usage in a human readable format."""
        from river import utils

        return utils.pretty.humanize_bytes(self._raw_memory_usage)


class Estimator(Base, abc.ABC):
    """An estimator."""

    @property
    def _supervised(self):
        """Indicates whether or not the estimator is supervised or not.
        This is useful internally for determining if an estimator expects to be provided with a `y`
        value in it's `learn_one` method. For instance we use this in a pipeline to know whether or
        not we should pass `y` to an estimator or not.
        """
        return True

    def __or__(self, other):
        """Merge with another Transformer into a Pipeline."""
        from river import compose

        if isinstance(other, compose.Pipeline):
            return other.__ror__(self)
        return compose.Pipeline(self, other)

    def __ror__(self, other):
        """Merge with another Transformer into a Pipeline."""
        from river import compose

        if isinstance(other, compose.Pipeline):
            return other.__or__(self)
        return compose.Pipeline(other, self)

    def _more_tags(self):
        return set()

    @property
    def _tags(self) -> typing.Dict[str, bool]:
        """Return the estimator's tags.
        Tags can be used to specify what kind of inputs an estimator is able to process. For
        instance, some estimators can handle text, whilst others don't. Inheriting from
        `base.Estimator` will imply a set of default tags which can be overridden by implementing
        the `_more_tags` property.
        TODO: this could be a cachedproperty.
        """

        tags = self._more_tags()

        for parent in self.__class__.__mro__:
            try:
                tags |= parent._more_tags(self)  # type: ignore
            except AttributeError:
                pass

        return tags

    @classmethod
    def _unit_test_params(self):
        """Indicates which parameters to use during unit testing.
        Most estimators have a default value for each of their parameters. However, in some cases,
        no default value is specified. This class method allows to circumvent this issue when the
        model has to be instantiated during unit testing.
        This can also be used to override default parameters that are computationally expensive,
        such as the number of base models in an ensemble.
        """
        yield {}

    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
        """
        return set()


class BaseTransformer:
    def __add__(self, other):
        """Fuses with another Transformer into a TransformerUnion."""
        from river import compose

        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(self, other)

    def __radd__(self, other):
        """Fuses with another Transformer into a TransformerUnion."""
        from river import compose

        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion(other, self)

    def __mul__(self, other):
        from river import compose

        if isinstance(other, (Transformer, compose.Pipeline)):
            return compose.TransformerProduct(self, other)

        return compose.Grouper(transformer=self, by=other)

    def __rmul__(self, other):
        """Creates a Grouper."""
        return self * other

    @abc.abstractmethod
    def transform_one(self, x: dict) -> dict:
        """Transform a set of features `x`.
        Parameters
        ----------
        x
            A dictionary of features.
        Returns
        -------
        The transformed values.
        """


class Transformer(Estimator, BaseTransformer):
    """A transformer."""

    @property
    def _supervised(self):
        return False

    def learn_one(self, x: dict) -> "Transformer":
        """Update with a set of features `x`.
        A lot of transformers don't actually have to do anything during the `learn_one` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_one` can override this
        method.
        Parameters
        ----------
        x
            A dictionary of features.
        Returns
        -------
        self
        """
        return self


class MiniBatchTransformer(Transformer):
    """A transform that can operate on mini-batches."""

    @abc.abstractmethod
    def transform_many(self, X: "pd.DataFrame") -> "pd.DataFrame":
        """Transform a mini-batch of features.
        Parameters
        ----------
        X
            A DataFrame of features.
        Returns
        -------
        A new DataFrame.
        """

    def learn_many(self, X: "pd.DataFrame") -> "Transformer":
        """Update with a mini-batch of features.
        A lot of transformers don't actually have to do anything during the `learn_many` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_many` can override this
        method.
        Parameters
        ----------
        X
            A DataFrame of features.
        Returns
        -------
        self
        """
        return self
