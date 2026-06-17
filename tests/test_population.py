# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Population evolutionary engine."""

from __future__ import annotations

import pytest

from genetic_algorithm import Population
from genetic_algorithm.plugins import max_value, negative_sphere, target_vector


def _pop(fitness_fn=max_value, **kwargs):
    defaults = {
        "pop_size": 10,
        "chromosome_size": 5,
        "bounds": [(0, 10)] * 5,
        "fitness_fn": fitness_fn,
        "seed": 36,
    }
    defaults.update(kwargs)
    return Population(**defaults)


def test_population_size_and_len():
    pop = _pop()
    assert len(pop) == 10
    assert len(pop.data) == 10


def test_calculate_fitness_sets_all():
    pop = _pop()
    pop.calculate_fitness()
    assert all(c.fitness is not None for c in pop.data)


def test_best_in_generation_is_sorted_desc():
    pop = _pop()
    pop.calculate_fitness()
    top = pop.best_in_generation(3)
    assert len(top) == 3
    assert top[0].fitness >= top[1].fitness >= top[2].fitness


def test_selection_elitist_returns_best():
    pop = _pop()
    pop.calculate_fitness()
    parents = pop.selection(num_parents=4, method="elitist")
    best = pop.best_in_generation(1)[0]
    assert parents[0].fitness == best.fitness


def test_selection_roulette_runs():
    pop = _pop()
    pop.calculate_fitness()
    parents = pop.selection(method="roulette")
    assert len(parents) >= 1


def test_selection_invalid_method_raises():
    pop = _pop()
    pop.calculate_fitness()
    with pytest.raises(ValueError):
        pop.selection(method="nope")


def test_roulette_handles_negative_fitness():
    # negative_sphere yields negative scores -> roulette must not crash.
    pop = _pop(fitness_fn=negative_sphere)
    pop.calculate_fitness()
    parents = pop.selection(method="roulette")
    assert len(parents) >= 1


@pytest.mark.parametrize("method,kwargs", [("single_point", {}), ("k_points", {"k": 2})])
def test_crossover_methods(method, kwargs):
    pop = _pop()
    pop.calculate_fitness()
    pop.selection(method="elitist")
    offspring = pop.crossover(method=method, **kwargs)
    assert len(offspring) >= 1


def test_crossover_invalid_method_raises():
    pop = _pop()
    pop.calculate_fitness()
    pop.selection(method="elitist")
    with pytest.raises(Exception, match="implement"):
        pop.crossover(method="nope")


def test_mutation_runs():
    pop = _pop()
    pop.calculate_fitness()
    pop.selection(method="elitist")
    pop.crossover(method="k_points", k=2)
    mutated = pop.mutation(method="probability_mutation")
    assert len(mutated) >= 1


def test_multi_threading_fitness():
    pop = _pop(multi_threading=True, max_workers=2)
    pop.calculate_fitness()
    assert all(c.fitness is not None for c in pop.data)


def test_seed_reproducibility():
    a = _pop(seed=123)
    b = _pop(seed=123)
    assert [c.data for c in a.data] == [c.data for c in b.data]


def test_full_loop_improves_fitness():
    target = [5.0, 5.0, 5.0, 5.0, 5.0]
    pop = Population(
        pop_size=30,
        chromosome_size=5,
        bounds=[(0, 10)] * 5,
        fitness_fn=target_vector(target),
        elitism=True,
        num_elitists=2,
        seed=36,
    )
    pop.calculate_fitness()
    first_best = pop.best_in_generation(1)[0].fitness

    last_best = first_best
    for _ in range(25):
        pop.calculate_fitness()
        last_best = pop.best_in_generation(1)[0].fitness
        pop.selection(method="elitist")
        pop.crossover(method="k_points", k=2)
        pop.mutation(method="probability_mutation")

    # Negated distance: closer to target => higher (less negative) fitness.
    assert last_best >= first_best


def test_str_and_repr_are_strings():
    pop = _pop()
    assert isinstance(str(pop), str)
    assert isinstance(repr(pop), str)
    assert str(pop) == repr(pop)


def test_elitism_getter_and_setter():
    pop = _pop(elitism=True)
    assert pop.elitism is True
    pop.elitism = False
    assert pop.elitism is False


def test_construct_without_seed():
    # Exercises the `seed is None` branch in the constructor.
    pop = _pop(seed=None)
    assert len(pop) == 10


def test_kpoints_crossover_with_three_points():
    # k=3 hits the even-index branch in the k-point crossover loop.
    pop = _pop(chromosome_size=8, bounds=[(0, 10)] * 8)
    pop.calculate_fitness()
    pop.selection(method="elitist")
    offspring = pop.crossover(method="k_points", k=3)
    assert len(offspring) >= 1
    assert all(len(c.data) == 8 for c in offspring)


def test_roulette_with_replacement_positive_fitness():
    # `replace=True` path of the roulette helper (not reachable via selection()).
    pop = _pop(fitness_fn=max_value)
    pop.calculate_fitness()
    parents = pop._Population__roulette_selection(4, replace=True)
    assert len(parents) == 4


def test_roulette_with_replacement_negative_fitness():
    # `replace=True` uniform-fallback path when fitness can be negative.
    pop = _pop(fitness_fn=negative_sphere)
    pop.calculate_fitness()
    parents = pop._Population__roulette_selection(4, replace=True)
    assert len(parents) == 4
