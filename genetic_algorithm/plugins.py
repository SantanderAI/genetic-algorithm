# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Built-in example fitness plugins.

These are intentionally tiny, dependency-free reference criteria. They exist to
demonstrate the plugin contract (``Sequence[float] -> float``, higher is better)
and to give the test-suite and examples something to optimise. Real problems
ship their own plugin -- that is the whole point of separating the engine from
the judgment.

Importing this module registers the named criteria in the
:mod:`genetic_algorithm.fitness` registry, so they can also be looked up by name
via :func:`genetic_algorithm.fitness.get_fitness`.
"""

from __future__ import annotations

from collections.abc import Sequence

from .fitness import FitnessFunction, Genes, register_fitness


@register_fitness("max_value")
def max_value(genes: Genes) -> float:
    """Reward larger genes: fitness is the sum of all genes.

    A trivial monotonic objective, useful as a smoke test for the engine.
    """
    return float(sum(genes))


@register_fitness("negative_sphere")
def negative_sphere(genes: Genes) -> float:
    """Negated sphere function centred at the origin (higher is better).

    The classic sphere benchmark ``sum(g**2)`` is a *minimisation* problem; the
    engine maximises, so we return its negative. The optimum is the all-zero
    candidate with fitness ``0.0``.
    """
    return -float(sum(g * g for g in genes))


def target_vector(target: Sequence[float]) -> FitnessFunction:
    """Build a criterion that rewards proximity to a ``target`` vector.

    This is a *factory*: it returns a configured :class:`FitnessFunction`. It
    shows how a plugin can carry its own parameters without the engine knowing
    anything about them. Fitness is the negative squared Euclidean distance to
    ``target`` (higher is better; optimum is ``target`` itself with ``0.0``).

    Parameters
    ----------
    target : Sequence[float]
        The vector candidates should converge toward.

    Returns
    -------
    FitnessFunction
        A criterion closed over ``target``.
    """
    reference = list(target)

    def _fitness(genes: Genes) -> float:
        return -float(sum((g - t) ** 2 for g, t in zip(genes, reference, strict=False)))

    return _fitness


def weighted_sum(weights: Sequence[float]) -> FitnessFunction:
    """Build a criterion that scores candidates by a weighted sum of genes.

    Demonstrates multi-objective-style composition: each gene contributes
    according to its ``weight``, so several concerns can be folded into one
    scalar score.

    Parameters
    ----------
    weights : Sequence[float]
        Per-gene weights. Positive weights reward larger genes, negative ones
        penalise them (e.g. "maximise quality, minimise cost").

    Returns
    -------
    FitnessFunction
        A criterion closed over ``weights``.
    """
    coeffs = list(weights)

    def _fitness(genes: Genes) -> float:
        return float(sum(w * g for w, g in zip(coeffs, genes, strict=False)))

    return _fitness
