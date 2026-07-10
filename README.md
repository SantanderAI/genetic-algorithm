# genetic-algorithm

> **Open source by Santander AI Lab.** A dependency-free Python **genetic-algorithm library / engine** with **pluggable fitness criteria** — the reusable search core for building an **LLM / AI autoresearcher** (generate → evaluate → select → repeat).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/SantanderAI/genetic-algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/SantanderAI/genetic-algorithm/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SantanderAI/genetic-algorithm/actions/workflows/codeql.yml/badge.svg)](https://github.com/SantanderAI/genetic-algorithm/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/SantanderAI/genetic-algorithm/badge)](https://scorecard.dev/viewer/?uri=github.com/SantanderAI/genetic-algorithm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196.svg)](https://conventionalcommits.org)

Part of [**Santander AI Open Source**](https://github.com/SantanderAI) — open source AI projects from Banco Santander ([santander.com](https://santander.com)).

`genetic-algorithm` is a tiny evolutionary engine — population, selection,
crossover, mutation — whose **fitness criterion is a swappable plugin**. The
engine never hard-codes *what* "better" means; it only ever asks a plugin for a
single number. That one design choice is what turns a textbook GA into the
**search core of an autoresearcher**.

## Why a GA with plugins is a door to better autoresearchers

A simple autoresearcher — the kind of loop Andrej Karpathy describes: *generate
a hypothesis → test it → measure it → keep the best → repeat* — is, structurally,
**an evolutionary loop**. A genetic algorithm with pluggable criteria gives you
exactly that machinery, ready-made, so you don't reimplement it every time.

### The loop and a GA are almost isomorphic

| Autoresearcher | Genetic algorithm |
|:---|:---|
| The candidates you explore (prompts, configs, hypotheses, strategies, code snippets) | **Population** |
| "Is this one better?" | **Fitness** — and this is where your **plugins** live |
| How the next batch of candidates is produced from the good ones | **Mutation / crossover** |
| Keep what works, drop what doesn't | **Selection** |

A naïve autoresearcher does a greedy or random search. The GA adds **structured
selective pressure plus diversity** — precisely what stops a self-reinforcing
LLM loop from collapsing into a single chain of reasoning and getting stuck in a
local optimum.

### Where the *plugin* part earns its keep

The bottleneck of any autoresearcher is **not the loop — it's defining "better"
well**. Making fitness a plugin is what makes that tractable:

- **Separate the engine from the judgment.** The engine doesn't need to know
  what you optimise; it needs a number. The plugin encapsulates the domain —
  summary quality, pipeline latency, test coverage, an experiment's score.
- **Compose multiple objectives.** Different plugins = different criteria you can
  combine ("maximise quality **and** minimise token cost").
- **Change problem without touching the core.** The same engine tunes prompts
  today and pipeline configs tomorrow — you only swap the plugin.
- **Put an LLM *inside* the criterion.** A plugin can be an *LLM-as-a-judge* that
  scores qualitative candidates where there is no obvious numeric metric.

See [`examples/autoresearcher.py`](examples/autoresearcher.py) for the mapping
made concrete, including the exact seam where a real LLM judge plugs in.

### Being honest about where it does *not* help

This is engineering, not a silver bullet:

- **GA is evaluation-hungry.** Every individual needs a fitness value. If
  evaluating means an LLM call or a real experiment, a population of 50 over 100
  generations is 5,000 calls — often prohibitive. For many problems a
  *hill-climber + LLM mutator* is cheaper and nearly as good.
- **The GA is only as good as the fitness plugin.** A badly designed criterion
  invites **reward hacking**: the loop finds candidates that score high and are
  junk. The plugin is the main point of failure, not the algorithm.
- **For small or convex search spaces, a GA is over-engineering.** Exhaustive or
  Bayesian search will win there.

### Why this specific combination is worth shipping

Where the tool beats "just a bare loop":

- **The bare loop is the proof of concept; GA + plugins is the productisable,
  reusable version.** You turn an ad-hoc script into infrastructure: a stable
  evolutionary engine with interchangeable criteria.
- **Diversity for free.** The GA keeps several lines of investigation alive in
  parallel instead of one fragile chain of thought.
- **Traceability.** Every generation is an auditable record of what was tried and
  why it survived — which fits a need for verifiable work.

## Installation

```bash
pip install -e .
```

The engine itself has **no third-party runtime dependencies** — it runs on the
Python standard library alone.

## Quick Start

```python
from genetic_algorithm import Population, register_fitness


# 1. Define the criterion — this is your plugin. Higher is better.
@register_fitness("max_ones")
def max_ones(genes):
    return float(sum(genes))


# 2. Hand the engine a population shape, bounds, and the plugin.
pop = Population(
    pop_size=30,
    chromosome_size=8,
    bounds=[(0, 1) for _ in range(8)],
    fitness_fn=max_ones,
    elitism=True,
    seed=36,
)

# 3. Run the loop: generate -> evaluate -> select -> recombine -> mutate.
for _ in range(25):
    pop.calculate_fitness()
    best = pop.best_in_generation(1)[0]
    pop.selection(method="roulette")
    pop.crossover(method="k_points", k=2)
    pop.mutation(method="probability_mutation")

print(best.data, best.fitness)
```

Any callable `Sequence[float] -> float` (higher is better) is a valid criterion,
so you can pass a function directly or register it by name for
configuration-driven runs (`get_fitness("max_ones")`).

## Examples

```bash
# Numeric optimisation toward a target vector
python -m examples.optimize_sphere

# The GA framed as a Karpathy-style autoresearcher (offline LLM-as-judge stub)
python -m examples.autoresearcher
```

## API at a glance

- `Population` — the evolutionary engine. Selection (`roulette`, `elitist`),
  crossover (`single_point`, `k_points`), mutation (`probability_mutation`,
  `twors`, `cim`, `thrors`), optional elitism, optional multi-threaded fitness
  evaluation (useful when the criterion is I/O-bound, e.g. an LLM judge), and a
  `seed` for reproducible runs.
- `Chromosome` — a single candidate: a list of float genes within per-gene
  bounds.
- `FitnessFunction` — the plugin contract (`Sequence[float] -> float`).
- `register_fitness` / `get_fitness` / `available_fitness` — a small registry for
  selecting criteria by name.
- `genetic_algorithm.plugins` — reference criteria: `max_value`,
  `negative_sphere`, and the `target_vector` / `weighted_sum` factories.

## Requirements

- **Python 3.10+**
- **No third-party runtime dependencies** — standard library only.
- Optional, for development only: `ruff`, `black`, `mypy`, `pytest`,
  `pytest-cov` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md)
and [Code of Conduct](CODE_OF_CONDUCT.md) before getting started.

- Report bugs and request features via [GitHub Issues](https://github.com/SantanderAI/genetic-algorithm/issues).
- External contributors sign the CLA (handled automatically by the CLA Assistant bot on your first PR).
- Run `ruff check .`, `black --check .`, `mypy genetic_algorithm`, and `pytest` before opening a PR.
- Keep the engine dependency-free (standard library only).

## Security

Please report security vulnerabilities responsibly. See our [Security Policy](.github/SECURITY.md)
for how to report (do **not** open a public issue for vulnerabilities). Contact:
**opensource@gruposantander.com** or use GitHub Security Advisories.

## Disclaimer

This software is an open source project from the **Santander AI Lab**, provided **"as is"** under its [license](LICENSE), without warranties or conditions of any kind. It is **not an official Banco Santander product or service**, carries no commitment of production support, and does not constitute financial, legal or professional advice.

"Santander" and its logo are registered trademarks of **Banco Santander, S.A.** The project license does not grant any right to use them beyond factual attribution.

If you believe you have found a security vulnerability, follow our [security policy](https://github.com/SantanderAI/.github/blob/main/SECURITY.md) — do not open a public issue. You are responsible for assessing the suitability of this software for your use case and for keeping your own deployments up to date.

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE)
and [NOTICE](NOTICE) files for details.

```
Copyright (c) 2026 Santander Group
SPDX-License-Identifier: Apache-2.0
```

## Citation

If you use `genetic-algorithm` in your research, please cite it:

```bibtex
@software{geneticalgorithm2026,
  author  = {{Santander AI Lab}},
  title   = {genetic-algorithm: a pluggable-fitness evolutionary engine},
  year    = {2026},
  url     = {https://github.com/SantanderAI/genetic-algorithm},
  license = {Apache-2.0}
}
```
