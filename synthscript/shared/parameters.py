from collections.abc import Callable, Sequence
from typing import Any, Generic, Literal, TypeVar

import numpy as np

Vector2D = np.ndarray[tuple[Literal[2], Any]]
RNGInput = np.random.Generator | int | None


def _make_rng(value: RNGInput) -> np.random.Generator:
    if isinstance(value, np.random.Generator):
        return value
    return np.random.default_rng(value)


class RandomizedParameter:
    __slots__ = ("_rng",)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: RNGInput) -> None:
        self._rng = _make_rng(value)


class Parameter(RandomizedParameter):
    """
    Represents a probability distribution or a single value.
    Useful for transformation configuration.
    """

    __slots__ = ("_bounds", "_value")

    def __init__(
        self,
        value: "Parameter | float | Callable[[], float]",
        *,
        rng: RNGInput = None,
    ):
        if isinstance(value, Parameter):
            self._value = getattr(value, "_value", value)
            self._bounds = value._bounds
            if rng is None:
                rng = value.rng
        elif isinstance(value, (float, int)):
            self._value = value
            self._bounds = (value, value)
        else:
            self._value = value
            self._bounds = (float("-inf"), float("inf"))
        self.rng = rng

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: RNGInput) -> None:
        self._rng = _make_rng(value)
        nested = getattr(self, "_value", None)
        if isinstance(nested, Parameter):
            nested.rng = self._rng

    def __call__(self) -> float:
        if isinstance(self._value, (float, int)):
            return self._value

        elif callable(self._value):
            return self._value()

        else:
            raise ValueError(
                f"Value is neither a callable nor a float or int, but {self._value}"
            )

    @property
    def bounds(self) -> tuple[float, float]:
        return self._bounds

    def is_bounded(self, low: float | None = None, high: float | None = None):

        return ((low is None) or (low <= self._bounds[0])) and (
            (high is None) or (high >= self._bounds[1])
        )


class NormalDistribution(Parameter):
    __slots__ = ("_mean", "_sigma")

    def __init__(
        self,
        mean: float = 0,
        sigma: float = 1,
        *,
        rng: RNGInput = None,
    ):
        self._mean = mean
        self._sigma = sigma
        self._bounds = (float("-inf"), float("inf"))
        self.rng = rng

    def __call__(self) -> float:
        return float(self._rng.normal(self._mean, self._sigma))

    def __repr__(self):
        return f"<N({self._mean},{self._sigma})>"


class TrimmedNormalDistribution(Parameter):
    __slots__ = ("_mean", "_sigma")

    def __init__(
        self,
        clip_low: float = -2,
        clip_high: float = 2,
        mean: float = 0,
        sigma: float = 1,
        *,
        rng: RNGInput = None,
    ):
        self._mean = mean
        self._sigma = sigma
        self._bounds = (clip_low, clip_high)
        self.rng = rng

    def __call__(self) -> float:
        m, M = self._bounds
        return min(max(m, float(self._rng.normal(self._mean, self._sigma))), M)

    def __repr__(self):
        return f"<TrimN({self._mean},{self._sigma})>"


class UniformDistribution(Parameter):
    __slots__ = ("_max", "_min")

    def __init__(self, low: float, high: float, *, rng: RNGInput = None):
        self._min = low
        self._max = high
        self._bounds = (low, high)
        self.rng = rng

    def __call__(self) -> float:
        return float(self._rng.uniform(self._min, self._max))

    def __repr__(self):
        return f"<U({self._min},{self._max})>"


T = TypeVar("T")


class DiscreteDistribution(RandomizedParameter, Generic[T]):
    def __init__(
        self,
        values: Sequence[T],
        probabilities: Sequence[float] | None,
        *,
        rng: RNGInput = None,
    ):
        if probabilities is not None:
            if len(probabilities) != len(values):
                raise ValueError(
                    "The given probabilities and values must have the same length"
                )
            if not (np.isclose(sum(probabilities), 1)):
                raise ValueError("The sum of the given probabilities must be 1.")
        self._values = values
        self._probabilities = (
            [1 / len(values)] * len(values) if probabilities is None else probabilities
        )
        self._bounds = None
        self.rng = rng

    def __call__(self) -> T:
        return self._rng.choice(
            self._values, p=self._probabilities
        )  # ty: ignore[no-matching-overload]
