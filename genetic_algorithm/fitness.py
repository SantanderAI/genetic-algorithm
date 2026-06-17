# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Pluggable fitness criteria for the genetic algorithm engine.

The engine never hard-codes *what* "better" means. A **fitness function** is any
callable that maps a candidate (a sequence of float genes) to a single ``float``
score where **higher is better**. That single number is the only contract
between the evolutionary engine and the problem domain.

This separation -- *engine* vs. *judgment* -- is what lets the same evolutionary
loop optimise prompts today and pipeline configs tomorrow, just by swapping the
plugin. It is also what makes the engine usable as the search core of an
autoresearcher: the plugin encapsulates the "is this candidate better?" question
(a metric, a test-suite score, or an LLM-as-a-judge), and the engine supplies the
generate -> evaluate -> select -> repeat loop.

Two ways to provide a criterion:

* Pass any callable ``Callable[[Sequence[float]], float]`` directly to
  :class:`~genetic_algorithm.population.Population`.
* Register a named plugin with :func:`register_fitness` and look it up later with
  :func:`get_fitness` -- handy for configuration-driven runs where the criterion
  is selected by name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, overload, runtime_checkable

#: A candidate solution: an ordered sequence of float-valued genes.
Genes = Sequence[float]


@runtime_checkable
class FitnessFunction(Protocol):
    """Structural type for a fitness criterion.

    Any callable that accepts the gene sequence of a candidate and returns a
    single ``float`` (higher is better) satisfies this protocol.
    """

    def __call__(self, genes: Genes) -> float:  # pragma: no cover - protocol
        ...


# Name -> criterion registry for configuration-driven / plugin-style lookup.
_REGISTRY: dict[str, FitnessFunction] = {}


@overload
def register_fitness(name: str) -> Callable[[FitnessFunction], FitnessFunction]: ...


@overload
def register_fitness(name: str, fn: FitnessFunction) -> FitnessFunction: ...


def register_fitness(
    name: str,
    fn: FitnessFunction | None = None,
) -> Callable[[FitnessFunction], FitnessFunction] | FitnessFunction:
    """Register a fitness criterion under ``name``.

    Can be used directly::

        register_fitness("max_value", max_value)

    or as a decorator::

        @register_fitness("max_value")
        def max_value(genes):
            return float(sum(genes))

    Parameters
    ----------
    name : str
        Unique key used to look the criterion up with :func:`get_fitness`.
    fn : FitnessFunction, optional
        The criterion to register. If omitted, a decorator is returned.

    Returns
    -------
    FitnessFunction or callable
        The registered function (direct form) or a decorator (decorator form).

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"A fitness function named {name!r} is already registered")

    def _register(func: FitnessFunction) -> FitnessFunction:
        _REGISTRY[name] = func
        return func

    if fn is None:
        return _register
    return _register(fn)


def get_fitness(name: str) -> FitnessFunction:
    """Return the registered fitness criterion named ``name``.

    Raises
    ------
    KeyError
        If no criterion is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"No fitness function registered as {name!r}. Available: {available}"
        ) from None


def available_fitness() -> list[str]:
    """Return the sorted names of all registered fitness criteria."""
    return sorted(_REGISTRY)
