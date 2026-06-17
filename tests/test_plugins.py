# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the built-in example fitness plugins."""

from __future__ import annotations

import pytest

from genetic_algorithm.plugins import (
    max_value,
    negative_sphere,
    target_vector,
    weighted_sum,
)


def test_max_value_sums_genes():
    assert max_value([1.0, 2.0, 3.0]) == 6.0


def test_negative_sphere_optimum_at_origin():
    assert negative_sphere([0.0, 0.0]) == 0.0
    assert negative_sphere([1.0, 0.0]) == pytest.approx(-1.0)
    # Moving away from the origin reduces fitness.
    assert negative_sphere([3.0]) < negative_sphere([1.0])


def test_target_vector_peaks_at_target():
    fit = target_vector([5.0, 5.0])
    assert fit([5.0, 5.0]) == 0.0
    assert fit([5.0, 5.0]) > fit([0.0, 0.0])


def test_weighted_sum_applies_weights():
    fit = weighted_sum([2.0, -1.0])
    assert fit([3.0, 4.0]) == pytest.approx(2.0 * 3.0 - 1.0 * 4.0)


def test_factories_return_callables():
    assert callable(target_vector([1.0]))
    assert callable(weighted_sum([1.0]))
