# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""The :class:`Population` -- the evolutionary engine.

``Population`` manages a list of
:class:`~genetic_algorithm.chromosome.Chromosome` candidates and drives the
generic genetic-algorithm loop: evaluate fitness, select parents, cross them
over, and mutate the offspring. The *judgment* of what "fit" means is injected
as a pluggable ``fitness_fn`` -- the engine itself is problem-agnostic, which is
exactly what makes it reusable as the search core of an autoresearcher.
"""

from __future__ import annotations

import random
from copy import deepcopy
from queue import Queue
from threading import Thread

from .chromosome import Chromosome
from .fitness import FitnessFunction


class Population:

    def __init__(
        self,
        pop_size: int,
        chromosome_size: int,
        bounds: list,
        fitness_fn: FitnessFunction,
        decimals=0,
        parents_size=0.5,
        offspring_size=0.5,
        multi_threading=False,
        max_workers=4,
        elitism=False,
        num_elitists=1,
        seed=None,
    ):
        """Main constructor of the Population class

        This class has the main goal to manage the population of `Chromosome`s
        and run the main functions in genetic algorithm (ga).

        Parameters
        ----------
        pop_size : int
            Defines the population size
        chromosome_size : int
            Defines the size (array length) of the `Chromosome`
        bounds : list[tuple]
            A `list` of `tuple`s to define the maximum and minimum value per gene
        fitness_fn : FitnessFunction
            Pluggable criterion mapping a candidate's genes to a single `float`
            score (higher is better). This is the only place the problem domain
            lives; the engine never inspects it.
        decimals : int, optional
            If you want a decimal `Chromosome` you should define it as numerical
            and specify the `decimal` witch every gene will be divided for
        parents_size :
        offspring_size
        multi_threading : bool, optional
            Default to False, defines if you want to run the fitness function in
            multi threading
        max_workers : int, optional
            Default to 4, defines how many workers does the multi-threading option
            will run
        seed : int, optional
            If given, seeds the standard-library RNG for reproducible runs.
        """
        if seed is not None:
            random.seed(seed)
        self._fitness_fn = fitness_fn
        self._pop_size = pop_size
        self._chromosome_size = chromosome_size
        self._bounds = bounds
        self._decimals = decimals
        self._parents_size = parents_size
        self._offspring_size = offspring_size
        self._multi_threading = multi_threading
        self.__generation_parents: list[Chromosome] = []
        self.elitism = elitism
        self.__num_elitists = num_elitists
        self.__gen_best_chromosomes: list[Chromosome] = []

        if multi_threading:
            self.__q: Queue = Queue()
            for _ in range(max_workers):
                t = Thread(target=self.__worker)
                t.daemon = True
                t.start()
        self.__data = self.__new_population()

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        """This property hold the population list of `Chromosome`s"""
        return self.__data

    @property
    def elitism(self):
        return self.__elitism

    @elitism.setter
    def elitism(self, value):
        self.__elitism = value

    def __new_population(self):
        """Initialize a new population (size is given by the class variable)

        Returns
        -------
        list [Chromosome]
            The list of chromosomes witch size is `self._pop_size`
        """
        data = []
        for _ in range(self._pop_size):
            data.append(Chromosome(self._chromosome_size, self._bounds, decimals=self._decimals))
        return data

    def __worker(self):
        while True:
            chromosome = self.__q.get()
            chromosome.calculate_fitness(self._fitness_fn)
            self.__q.task_done()

    def best_in_generation(self, n=1):
        """Calculate the best Chromosome (in fitness terms) of the current population

        Parameters
        ----------
        n : int
            The number of top Chromosomes selected from the population

        Returns
        -------
        list[Chromosome]
            The reference of the best chromosome in the current population
        """
        return [
            deepcopy(chromosome)
            for chromosome in sorted(self.data, key=(lambda x: x.fitness), reverse=True)[:n]
        ]

    def calculate_fitness(self):
        """

        Returns
        -------

        """
        for chromosome in self.data:
            # If chromosome fitness is calculated, dont do it again
            # TODO:: review this the fitness function
            # if chromosome.fitness:
            #     continue
            # Calculate chromosome fitness in multi or single thread
            if self._multi_threading:
                self.__q.put(chromosome)
            else:
                chromosome.calculate_fitness(self._fitness_fn)

        # If multi-threading, join the threads
        if self._multi_threading:
            self.__q.join()

    def __roulette_selection(self, num_parents, replace=False):
        """Selection function based in roulette idea

        The main idea in this algorithm is that the chromosomes with the higher
        fitness have a higher probability to be selected for the mating pool

        Parameters
        ----------
        num_parents : int
            The number of desired parents to be selected from the population
        replace : boolean, optional
            Select if there will be replacement or not in the selection

        Returns
        -------
        list [Chromosome]
            The list of parents (chromosomes) that have been selected for the
            mating pool.
        """
        fitnesses = [c.fitness for c in self.data]
        fitness_sum = sum(fitnesses)
        # Roulette selection assumes non-negative fitness. When the active
        # criterion can return negative scores (e.g. a negated-distance
        # plugin), fall back to uniform sampling so the engine still makes
        # progress; use `method='elitist'` for those problems.
        if min(fitnesses) < 0 or fitness_sum <= 0:
            population = list(self.data)
            if replace:
                return [random.choice(population) for _ in range(num_parents)]
            return random.sample(population, min(num_parents, len(population)))
        weights = [f / fitness_sum for f in fitnesses]
        if replace:
            return random.choices(self.data, weights=weights, k=num_parents)
        return self.__weighted_sample_without_replacement(weights, num_parents)

    def __weighted_sample_without_replacement(self, weights, k):
        """Sample `k` distinct chromosomes with probability proportional to weight"""
        items = list(self.data)
        w = list(weights)
        k = min(k, len(items))
        chosen = []
        for _ in range(k):
            idx = random.choices(range(len(items)), weights=w, k=1)[0]
            chosen.append(items.pop(idx))
            w.pop(idx)
        return chosen

    def __elitist_selection(self, num_parents):
        """Selection function based in elitism idea

        Select the top `num_parents` `Chromosome`s from the population based on the
        fitness

        Parameters
        ----------
        num_parents : int
            The number of desired parents to be selected from the population

        Returns
        -------
        list [Chromosome]
            The list of parents (chromosomes) that have been selected for the
            mating pool.

        """
        pop_sorted = sorted(self.data, key=lambda chromosome: chromosome.fitness, reverse=True)
        return pop_sorted[:num_parents]

    def selection(self, num_parents=None, method="roulette"):
        """Perform the population selection for *mating pool*

        Parameters
        ----------
        num_parents : int
            The number of desired parents to be selected from the population
        method : {'roulette', 'elitist'}, optional
            Defines the kind of selection to be performed by the genetic algorithm

        Returns
        -------
        list [Chromosome]
            The list of parents (`Chromosome`) that have been selected for the
            mating pool.
        """
        num_parents = num_parents if num_parents else round(len(self.__data) / 2)
        if method == "roulette":
            parents = self.__roulette_selection(num_parents)
        elif method == "elitist":
            parents = self.__elitist_selection(num_parents)
        else:
            raise ValueError(
                "Population.selection only accepts `methods` available: "
                + "{roulette, elitist}. Please, read the docs for more info"
            )

        self.__generation_parents = parents
        return parents

    def __single_point_crossover(self, parent1, parent2):
        """Crossover with single cut points

        Parameters
        ----------
        parent1 : Chromosome
            The first parent for mating
        parent2 : Chromosome
            Second parent for mating

        Returns
        -------
        (Chromosome , Chromosome)
            The child chromosome generated by the two parents
        """
        child1 = Chromosome(self._chromosome_size, self._bounds, decimals=self._decimals)
        child2 = Chromosome(self._chromosome_size, self._bounds, decimals=self._decimals)
        point = random.randrange(0, self._chromosome_size)

        child1.data[:point] = parent1.data[:point]
        child1.data[point:] = parent2.data[point:]

        child2.data[:point] = parent2.data[:point]
        child2.data[point:] = parent1.data[point:]

        return child1, child2

    def __k_point_crossover(self, parent1, parent2, k):
        """Crossover with multiple cut points

        Parameters
        ----------
        parent1 : Chromosome
            The first parent for mating
        parent2 : Chromosome
            Second parent for mating
        k : int
            Number of cuts

        Returns
        -------
        (Chromosome, Chromosome)
            The child chromosome generated by the two parents
        """
        assert k >= 2
        child1 = Chromosome(self._chromosome_size, self._bounds, decimals=self._decimals)
        child2 = Chromosome(self._chromosome_size, self._bounds, decimals=self._decimals)
        points = sorted(random.sample(range(self._chromosome_size), k))

        # Take the first cut of each parent
        child1.data[: points[0]] = parent1.data[: points[0]]
        child2.data[: points[0]] = parent2.data[: points[0]]

        prev_p = points[0]
        # Iterate over points to mix parents genes into children
        for i, p in enumerate(points[1:]):
            if (i + 1) % 2 == 0:
                child1.data[prev_p:p] = parent1.data[prev_p:p]
                child2.data[prev_p:p] = parent2.data[prev_p:p]
            else:
                child1.data[prev_p:p] = parent2.data[prev_p:p]
                child2.data[prev_p:p] = parent1.data[prev_p:p]
            prev_p = p

        # Add the final cut of each parent to their children
        child1.data[points[-1] :] = (
            parent1.data[points[-1] :] if len(points) % 2 == 0 else parent2.data[points[-1] :]
        )
        child2.data[points[-1] :] = (
            parent2.data[points[-1] :] if len(points) % 2 == 0 else parent1.data[points[-1] :]
        )

        return child1, child2

    def crossover(self, children_size=None, method="k_points", k=2):
        """

        Calculate the new children population based on the parents from the selection
        process. The num of iterations is divided by two because each pair of parents
        will produce a new pair of children

        Parameters
        ----------
        children_size : int, optional
            The number of children to be returned by the function. Default will
            return a population of `self._pop_size` children
        method : {'single_point', 'k_point'}, optional
            The crossover method to be performed by the algorithm
        k : int, optional
            In the case of k_point crossover method, this is the number of cuts
            to be made by the algorithm. Default to two (2)

        Returns
        -------
        list [Chromosome]
            Return the offspring made by `parents` crossover. This new population
            will be part of the next generation of the genetic algorithm

        """
        children_size = (
            round(len(self.__data) / 2) if not children_size else round(children_size / 2)
        )
        if self.__elitism:
            self.__gen_best_chromosomes = self.best_in_generation(self.__num_elitists)
            children_size = children_size - self.__num_elitists

        offspring = []

        # Pick two parents at random from the mating pool and perform the crossover
        for _ in range(children_size):
            parent1, parent2 = random.sample(list(self.__generation_parents), 2)
            if method == "single_point":
                children = self.__single_point_crossover(parent1, parent2)
            elif method == "k_points":
                children = self.__k_point_crossover(parent1, parent2, k)
            else:
                raise Exception(
                    "Crossover method not not implement. You should consider "
                    + "contributing the project with adding the implementation"
                )
            # Append the new children to the offspring population
            offspring += children

        self.__data = offspring
        # If elitism, add the last generation best_chromosomes to the current one to be mutated
        if self.__elitism:
            self.__data = self.__data + [
                deepcopy(chromosome) for chromosome in self.__gen_best_chromosomes
            ]
        return offspring

    def mutation(self, method="probability_mutation", mutation_prob=None):
        """This method mutates all of Chromosomes in the current population

        This mutation is managed by `Chromosome` class and we can decide each
        generation witch mutation operator we want to use. This is usefull
        beacuse we can explore more at the begining and converge more at the
        end.

        If `self.elitism` is set, the number of selected elitist will be
        added to the population after this phase. The idea is to preserve
        those *best* chromosomes as they are in the future generations so
        they can crossover again.

        Parameters
        ----------
        method : str, optional
            Defines the method that `Chromosome` will use to mutate the genes.
        mutation_prob : float, optional
            In the case of probability based *mutation operator*, this parameter
            defines the threshold probability to mutate each gene.

        Returns
        -------
        list [Chromosome]
            Returns the final population of this generation mutated by `method`
            `Chromosome` operator.

        """
        for chromosome in self.data:
            chromosome.mutate(method=method, mutation_prob=mutation_prob)
        # If elitism, add the best chromosomes in the last generation to the new one
        if self.__elitism:
            self.__data = self.__data + self.__gen_best_chromosomes
        return self.data
