# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""The :class:`Chromosome` -- a single candidate solution.

A chromosome is an ordered list of float-valued genes within per-gene
``bounds``. It knows how to initialise itself, evaluate itself against a
*pluggable* fitness criterion, and mutate via one of several operators. It
holds **no** problem-specific logic: *what* a good chromosome is comes entirely
from the fitness function supplied by the caller.
"""

from __future__ import annotations

import random
from uuid import uuid4

from .fitness import FitnessFunction


class Chromosome:
    def __init__(self, chromosome_size: int, bounds: list, data=None, decimals=0):
        """Main constructor of the Chromosome class

        Parameters
        ----------
        chromosome_size: int
            T
        bounds : list[tuple]
            A list of tuples with the maximum and minimum per gene values
        data : list[float]
            The initial data of the Chromosome
        decimals : int, optional

        """
        self.__data = data
        self._chromosome_size = chromosome_size
        self._bounds = bounds
        self._decimals = decimals
        self.__fitness: float | None = None
        self.__id = str(uuid4())

        if not data:
            self._init_chromosome()

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    @property
    def data(self):
        return self.__data

    @property
    def id(self):
        return self.__id

    @property
    def fitness(self):
        return self.__fitness

    def __generate_gene(self, i):
        """Generate a new gene having `self._bounds` into account

        Parameters
        ----------
        i : int
            The gene position that you want to generate

        Returns
        -------
        float
            The gene value

        """
        gene = random.randint(self._bounds[i][0], self._bounds[i][1])
        gene = gene / 10**self._decimals if self._decimals else gene
        return gene

    def _init_chromosome(self):
        """Initialize the Chromosome genes values

        Returns
        -------
        list [int]
            A list of genes (float) values of the chromosome

        """
        data = []
        for i in range(self._chromosome_size):
            gene = self.__generate_gene(i)
            data.append(gene)

        self.__data = data
        return self.data

    def calculate_fitness(self, fitness_fn: FitnessFunction) -> float:
        """Evaluate this chromosome against a pluggable fitness criterion

        The engine stays agnostic to *what* is being optimised: this callable
        is the only place the problem domain lives. Swap the plugin and the
        same chromosome can score a prompt, a config, or an experiment.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Any callable mapping the gene sequence to a single ``float`` score
            (higher is better).

        Returns
        -------
        float
            The fitness value of the Chromosome itself (also cached).

        """
        self.__fitness = float(fitness_fn(self.data))
        return self.fitness

    def __twors_mutate(self):
        """TWORDS method for mutation

        Twors mutation allows the exchange of position of two genes randomly chosen.
        Randomly pick two positions and exchange the genes.

        """
        p1, p2 = random.sample(range(len(self.data)), 2)
        saved_gene = self.data[p1]
        self.data[p1] = self.data[p2]
        self.data[p2] = saved_gene

        return self.data

    def __cim_mutate(self):
        """Centre Inverse Mutation

        Given a random point `p` in the chromosome_size range, the function inverse
        both sides of the chromosome (the genes order).
        The chromosome is divided into two sections. All genes in each section are copied
        and then inversely placed in the same section of a child.

        """
        p = random.randrange(0, len(self.data))
        self.__data = list(reversed(self.data[:p])) + list(reversed(self.data[p:]))

        return self.data

    def __thrors_mutation(self):
        """Thrors method for mutation

        Three genes are chosen randomly which shall take the different positions not
        necessarily successive i < j < l. the gene of the position i becomes in the position j
        and the one who was at this position will take the position l and the gene that has
        held this position takes the position i.

        """
        p1, p2, p3 = sorted(random.sample(range(len(self.data)), 3))
        p2_gene = self.data[p2]
        p3_gene = self.data[p3]

        self.__data[p2] = self.data[p1]
        self.__data[p3] = p2_gene
        self.__data[p1] = p3_gene

        return self.data

    def __probability_mutation(self, probability):
        for i, _ in enumerate(self.data):
            if random.random() > probability:
                continue
            self.__data[i] = self.__generate_gene(i)
        return self.data

    def mutate(self, method="probability_mutation", mutation_prob=None):
        if method == "probability_mutation":
            mutation_prob = mutation_prob if mutation_prob else 1 / len(self.data)
            return self.__probability_mutation(mutation_prob)
        elif method == "twors":
            return self.__twors_mutate()
        elif method == "cim":
            return self.__cim_mutate()
        elif method == "thrors":
            return self.__thrors_mutation()
        else:
            raise ValueError("`Method` selected is not available")
