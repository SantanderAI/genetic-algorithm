# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Pluggable fitness criteria** (`genetic_algorithm.fitness`): a
  `FitnessFunction` protocol plus a name registry (`register_fitness`,
  `get_fitness`, `available_fitness`) so the engine is fully decoupled from the
  problem domain.
- Built-in example fitness plugins (`genetic_algorithm.plugins`): `max_value`,
  `negative_sphere`, and the `target_vector` / `weighted_sum` factories.
- Runnable examples: `examples/optimize_sphere.py` and
  `examples/autoresearcher.py` (the genetic algorithm framed as the search core
  of a Karpathy-style autoresearcher, with an LLM-as-a-judge seam).
- Test suite (`tests/`) covering the engine, chromosome, registry, and plugins.
- Open-source readiness scaffolding: `NOTICE`, `CONTRIBUTING.md` (CLA),
  `CODE_OF_CONDUCT.md`, `.github/SECURITY.md`, `CODEOWNERS`, issue/PR templates,
  `CITATION.cff`, `pyproject.toml` tooling (ruff/black/mypy/pytest/coverage),
  SPDX headers, `.github/dependabot.yml`, and GitHub Actions workflows
  (`ci`, `codeql`, `dep-scan`, `license-check`, `pattern-check`, `scorecard`,
  `cla`, `stale`, `release`) with third-party actions pinned to SHA digests.

### Changed
- **Engine is now dependency-free**: replaced `numpy`/`pandas` usage with the
  Python standard library (`random`).
- `Population` now requires a `fitness_fn` and accepts an optional `seed` for
  reproducible runs; `Chromosome.calculate_fitness(fitness_fn)` takes the
  criterion as an argument.

### Removed
- `matrix-ga.py` example and its hard-coded, domain-specific fitness (a Parquet
  matrix scoring routine), replaced by the pluggable design and the new
  `examples/`.

### Fixed
- Elitist selection now returns the **best** chromosomes (previously sorted
  ascending and returned the worst).

## [0.1.0] - 2026-06-17

### Added
- Initial public release of the `genetic_algorithm` engine: `Chromosome` and
  `Population` with roulette / elitist selection, single- and k-point crossover,
  and probability / twors / cim / thrors mutation operators.

[Unreleased]: https://github.com/SantanderAI/genetic-algorithm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SantanderAI/genetic-algorithm/releases/tag/v0.1.0
