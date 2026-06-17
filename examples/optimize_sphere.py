# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Minimal numeric optimisation example.

Evolves a population of 6-gene candidates toward a fixed target vector using a
built-in plugin (``target_vector``). Because that criterion returns negative
scores (negated distance, higher is better), we use elitist selection, which
works regardless of the sign of the fitness.

Run from the repository root::

    python -m examples.optimize_sphere
    # or
    python examples/optimize_sphere.py
"""

from __future__ import annotations

from genetic_algorithm import Population
from genetic_algorithm.plugins import target_vector


def main() -> None:
    target = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    bounds = [(0, 100) for _ in target]

    pop = Population(
        pop_size=30,
        chromosome_size=len(target),
        bounds=bounds,
        fitness_fn=target_vector(target),
        elitism=True,
        num_elitists=2,
        seed=36,
    )

    best = None
    for generation in range(40):
        pop.calculate_fitness()
        best = pop.best_in_generation(1)[0]
        print(f"gen {generation:2d}  best fitness={best.fitness:10.2f}  {best.data}")

        pop.selection(method="elitist")
        pop.crossover(method="k_points", k=2)
        pop.mutation(method="probability_mutation")

    print(f"\nTarget : {target}")
    print(f"Best   : {best.data if best else None}")


if __name__ == "__main__":
    main()
