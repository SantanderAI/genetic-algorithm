# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Chromosome candidate model."""

from __future__ import annotations

import random

import pytest

from genetic_algorithm import Chromosome
from genetic_algorithm.plugins import max_value


def _make(size=5, low=0, high=10, decimals=0):
    bounds = [(low, high) for _ in range(size)]
    return Chromosome(size, bounds, decimals=decimals)


def test_init_length_and_bounds():
    c = _make(size=6, low=2, high=8)
    assert len(c.data) == 6
    assert all(2 <= g <= 8 for g in c.data)
    assert c.id  # a uuid string
    assert c.fitness is None


def test_decimals_scaling():
    c = _make(size=4, low=10, high=90, decimals=2)
    assert all(0.10 <= g <= 0.90 for g in c.data)


def test_explicit_data_is_preserved():
    c = Chromosome(3, [(0, 10)] * 3, data=[1.0, 2.0, 3.0])
    assert c.data == [1.0, 2.0, 3.0]


def test_calculate_fitness_uses_plugin():
    c = Chromosome(3, [(0, 10)] * 3, data=[1.0, 2.0, 3.0])
    assert c.calculate_fitness(max_value) == 6.0
    assert c.fitness == 6.0


def test_str_and_repr():
    c = Chromosome(2, [(0, 1)] * 2, data=[0.0, 1.0])
    assert str(c) == "[0.0, 1.0]"
    assert repr(c) == "[0.0, 1.0]"


@pytest.mark.parametrize("method", ["probability_mutation", "twors", "cim", "thrors"])
def test_mutation_methods_preserve_length(method):
    random.seed(1)
    c = _make(size=6)
    out = c.mutate(method=method, mutation_prob=1.0)
    assert len(out) == 6


def test_probability_mutation_allows_zero_probability():
    random.seed(1)
    c = Chromosome(3, [(0, 0)] * 3, data=[1.0, 2.0, 3.0])
    out = c.mutate(method="probability_mutation", mutation_prob=0)
    assert out == [1.0, 2.0, 3.0]


def test_permutation_mutations_preserve_multiset():
    c = Chromosome(5, [(0, 100)] * 5, data=[1.0, 2.0, 3.0, 4.0, 5.0])
    random.seed(2)
    c.mutate(method="twors")
    assert sorted(c.data) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_invalid_mutation_method_raises():
    c = _make()
    with pytest.raises(ValueError):
        c.mutate(method="does-not-exist")
