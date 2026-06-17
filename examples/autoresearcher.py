# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""A genetic algorithm as the search core of an autoresearcher.

Karpathy's "autoresearch" loop -- generate hypotheses, test them, measure, keep
the best, repeat -- is, structurally, an evolutionary loop. This example wires
that mapping explicitly:

* **Population**  -> the set of candidate research configurations being explored.
* **Fitness plugin** -> the *judge*: the "is this candidate better?" criterion.
* **Crossover / mutation** -> how the next batch of candidates is generated.
* **Selection** -> keeping what works and discarding what does not.

The bottleneck of any autoresearcher is **defining "better"**, not the loop. By
moving the judgment into a swappable plugin, the same engine can score prompts,
configs, or experiments -- and the judge can even be an LLM.

Here the judge is a deterministic offline stub so the example runs with no
network and no dependencies. The :func:`llm_judge_template` function shows the
exact seam where a real LLM-as-a-judge call would go.

Run from the repository root::

    python -m examples.autoresearcher
    # or
    python examples/autoresearcher.py
"""

from __future__ import annotations

from collections.abc import Sequence

from genetic_algorithm import Population, register_fitness

# A hidden "ideal configuration" the judge secretly rewards. In a real
# autoresearcher this is unknown -- the judge only ever returns a score, and the
# engine has to discover good candidates through selection pressure.
_IDEAL = [0.8, 0.2, 0.6, 0.9, 0.1]


@register_fitness("research_judge")
def research_judge(genes: Sequence[float]) -> float:
    """Offline stand-in for an LLM-as-a-judge.

    Scores a candidate configuration on a 0..100 quality scale (higher is
    better). The score is a smooth, deterministic function of how close the
    candidate is to a hidden ideal, so the example converges without any
    external service.

    To use a *real* judge, replace the body with a call like the one sketched in
    :func:`llm_judge_template`: render the candidate into a prompt, ask a model
    to rate it, and parse a numeric score back out. The engine never changes --
    only this plugin does.
    """
    distance = sum((g - ideal) ** 2 for g, ideal in zip(genes, _IDEAL, strict=False))
    # Map squared distance (0 = perfect) to a bounded quality score.
    return 100.0 / (1.0 + distance)


def llm_judge_template(genes: Sequence[float]) -> float:  # pragma: no cover
    """Illustrative seam for an LLM-as-a-judge fitness plugin (not executed).

    This is intentionally not wired up; it documents the shape of a real judge::

        prompt = render_candidate_prompt(genes)            # candidate -> text
        reply = my_llm_client.score(prompt)                # one model call
        return parse_score(reply)                          # text -> float

    Because it satisfies the same ``Sequence[float] -> float`` contract, it can
    be dropped into :class:`~genetic_algorithm.Population` with no other change.
    """
    raise NotImplementedError("Wire up your own LLM client to use a real judge.")


def main() -> None:
    knobs = len(_IDEAL)
    # Genes are scaled 0..100 then divided by 100 (decimals=2) -> 0.00..1.00.
    bounds = [(0, 100) for _ in range(knobs)]

    pop = Population(
        pop_size=24,
        chromosome_size=knobs,
        bounds=bounds,
        fitness_fn=research_judge,
        decimals=2,
        elitism=True,
        num_elitists=2,
        seed=7,
    )

    best = None
    for generation in range(30):
        pop.calculate_fitness()  # "test + measure" every hypothesis
        best = pop.best_in_generation(1)[0]  # keep the best
        print(f"gen {generation:2d}  judge score={best.fitness:6.2f}  config={best.data}")

        pop.selection(method="roulette")  # selection pressure
        pop.crossover(method="k_points", k=2)  # recombine the survivors
        pop.mutation(method="probability_mutation")  # explore nearby variants

    print(f"\nDiscovered config : {best.data if best else None}")
    print(f"Judge's hidden ideal: {_IDEAL}")


if __name__ == "__main__":
    main()
