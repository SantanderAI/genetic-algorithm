# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pluggable fitness registry and protocol."""

from __future__ import annotations

import pytest

from genetic_algorithm import (
    FitnessFunction,
    available_fitness,
    get_fitness,
    register_fitness,
)


def test_register_and_get_direct():
    def crit(genes):
        return float(sum(genes))

    register_fitness("unit_direct", crit)
    assert get_fitness("unit_direct") is crit
    assert get_fitness("unit_direct")([1, 2, 3]) == 6.0


def test_register_as_decorator():
    @register_fitness("unit_decorator")
    def crit(genes):
        return -1.0

    assert get_fitness("unit_decorator") is crit
    assert "unit_decorator" in available_fitness()


def test_duplicate_name_raises():
    register_fitness("unit_dup", lambda g: 0.0)
    with pytest.raises(ValueError):
        register_fitness("unit_dup", lambda g: 1.0)


def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_fitness("definitely-not-registered")


def test_builtins_registered():
    names = available_fitness()
    assert "max_value" in names
    assert "negative_sphere" in names


def test_protocol_is_runtime_checkable():
    assert isinstance((lambda genes: 1.0), FitnessFunction)
