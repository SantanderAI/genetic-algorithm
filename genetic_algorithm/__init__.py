# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""A small, dependency-free genetic-algorithm engine with pluggable fitness.

The engine (:class:`Population` / :class:`Chromosome`) supplies the evolutionary
loop -- generate, evaluate, select, recombine, mutate -- while the *judgment* of
what "fit" means is injected as a pluggable :data:`FitnessFunction`. Separating
the engine from the criterion is what lets the same loop power very different
searches, and what makes it usable as the search core of an autoresearcher.
"""

from . import plugins  # noqa: F401  (registers the built-in example criteria)
from .chromosome import Chromosome
from .fitness import (
    FitnessFunction,
    available_fitness,
    get_fitness,
    register_fitness,
)
from .population import Population

__all__ = [
    "Population",
    "Chromosome",
    "FitnessFunction",
    "register_fitness",
    "get_fitness",
    "available_fitness",
]
