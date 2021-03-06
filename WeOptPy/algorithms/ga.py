# encoding=utf8

"""Genetic algorithm module."""

import numpy as np
from numpy import random as rand

from WeOptPy.algorithms.interfaces.algorithm import Algorithm
from WeOptPy.algorithms.interfaces.individual import (
	Individual,
	default_individual_init
)

__all__ = [
	'GeneticAlgorithm',
	'tournament_selection',
	'roulette_selection',
	'two_point_crossover',
	'multi_point_crossover',
	'uniform_crossover',
	'uniform_mutation',
	'creep_mutation',
	'crossover_uros',
	'mutation_uros'
]


def tournament_selection(pop, ic, ts, x_b, rnd=rand):
	r"""Tournament selection method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual in population.
		ts (int): Tournament size.
		x_b (Individual): Global best individual.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		Individual: Winner of the tournament.
	"""
	comps = [pop[i] for i in rand.choice(len(pop), ts, replace=False)]
	return comps[np.argmin([c.f for c in comps])]


def roulette_selection(pop, ic, ts, x_b, rnd=rand):
	r"""Roulette selection method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual in population.
		ts (int): Unused argument.
		x_b (Individual): Global best individual.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		Individual: selected individual.
	"""
	f = np.sum([x.f for x in pop])
	qi = np.sum([pop[i].f / f for i in range(ic + 1)])
	return pop[ic].x if rnd.rand() < qi else x_b


def two_point_crossover(pop, ic, cr, rnd=rand):
	r"""Two point crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	r = np.sort(rnd.choice(len(pop[ic]), 2))
	x = pop[ic].x
	x[r[0]:r[1]] = pop[io].x[r[0]:r[1]]
	return np.asarray(x)


def multi_point_crossover(pop, ic, n, rnd=rand):
	r"""Multi point crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		n (flat): TODO.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	r, x = np.sort(rnd.choice(len(pop[ic]), 2 * n)), pop[ic].x
	for i in range(n): x[r[2 * i]:r[2 * i + 1]] = pop[io].x[r[2 * i]:r[2 * i + 1]]
	return np.asarray(x)


def uniform_crossover(pop, ic, cr, rnd=rand):
	r"""Uniform crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	j = rnd.randint(len(pop[ic]))
	x = [pop[io][i] if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return np.asarray(x)


def crossover_uros(pop, ic, cr, rnd=rand):
	r"""Crossover made by Uros Mlakar.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	alpha = cr + (1 + 2 * cr) * rnd.rand(len(pop[ic]))
	x = alpha * pop[ic] + (1 - alpha) * pop[io]
	return x


def uniform_mutation(pop, ic, mr, task, rnd=rand):
	r"""Uniform mutation method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		mr (float): Mutation probability.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	j = rnd.randint(task.D)
	nx = [rnd.uniform(task.Lower[i], task.Upper[i]) if rnd.rand() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return np.asarray(nx)


def mutation_uros(pop, ic, mr, task, rnd=rand):
	r"""Mutation method made by Uros Mlakar.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual.
		mr (float): Mutation rate.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	return np.fmin(np.fmax(rnd.normal(pop[ic], mr * task.bRange), task.Lower), task.Upper)


def creep_mutation(pop, ic, mr, task, rnd=rand):
	r"""Creep mutation method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		mr (float): Mutation probability.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	ic, j = rnd.randint(len(pop)), rnd.randint(task.D)
	nx = [rnd.uniform(task.lower[i], task.upper[i]) if rnd.rand() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return np.asarray(nx)


class GeneticAlgorithm(Algorithm):
	r"""Implementation of Genetic algorithm.

	Algorithm:
		Genetic algorithm

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		Ts (int): Tournament size.
		Mr (float): Mutation rate.
		Cr (float): Crossover rate.
		Selection (Callable[[numpy.ndarray[Individual], int, int, Individual, mtrand.RandomState], Individual]): Selection operator.
		Crossover (Callable[[numpy.ndarray[Individual], int, float, mtrand.RandomState], Individual]): Crossover operator.
		Mutation (Callable[[numpy.ndarray[Individual], int, float, Task, mtrand.RandomState], Individual]): Mutation operator.

	See Also:
		* :class:`WeOptPy.algorithms.Algorithm`
	"""
	Name = ['GeneticAlgorithm', 'GA']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.
		"""
		return r"""On info"""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* Ts (Callable[[int], bool]): Tournament size.
				* Mr (Callable[[float], bool]): Probability of mutation.
				* Cr (Callable[[float], bool]): Probability of crossover.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'Ts': lambda x: isinstance(x, int) and x > 1,
			'Mr': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'Cr': lambda x: isinstance(x, float) and 0 <= x <= 1
		})
		return d

	def set_parameters(self, n=25, ts=5, mr=0.25, cr=0.25, selection=tournament_selection, crossover=uniform_crossover, mutation=uniform_mutation, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			n (Optional[int]): Population size.
			ts (Optional[int]): Tournament selection.
			mr (Optional[int]): Mutation rate.
			cr (Optional[float]): Crossover rate.
			selection (Optional[Callable[[numpy.ndarray[Individual], int, int, Individual, mtrand.RandomState], Individual]]): Selection operator.
			crossover (Optional[Callable[[numpy.ndarray[Individual], int, float, mtrand.RandomState], Individual]]): Crossover operator.
			mutation (Optional[Callable[[numpy.ndarray[Individual], int, float, Task, mtrand.RandomState], Individual]]): Mutation operator.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.setParameters`
			* Selection:
				* :func:`WeOptPy.algorithms.TournamentSelection`
				* :func:`WeOptPy.algorithms.RouletteSelection`
			* Crossover:
				* :func:`WeOptPy.algorithms.UniformCrossover`
				* :func:`WeOptPy.algorithms.TwoPointCrossover`
				* :func:`WeOptPy.algorithms.MultiPointCrossover`
				* :func:`WeOptPy.algorithms.CrossoverUros`
			* Mutations:
				* :func:`WeOptPy.algorithms.UniformMutation`
				* :func:`WeOptPy.algorithms.CreepMutation`
				* :func:`WeOptPy.algorithms.MutationUros`
		"""
		Algorithm.set_parameters(self, n=n, itype=ukwargs.pop('itype', Individual), init_pop_func=ukwargs.pop('init_pop_func', default_individual_init), **ukwargs)
		self.Ts, self.Mr, self.Cr = ts, mr, cr
		self.Selection, self.Crossover, self.Mutation = selection, crossover, mutation

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **dparams):
		r"""Core function of GeneticAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals function/fitness value.
			args (lst): Additional argument.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New populations function/fitness values.\
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Additional arguments.
				6. Additional keyword arguments.
		"""
		npop = np.empty(self.NP, dtype=object)
		for i in range(self.NP):
			ind = self.itype(x=self.Selection(pop, i, self.Ts, xb, self.Rand), e=False)
			ind.x = self.Crossover(pop, i, self.Cr, self.Rand)
			ind.x = self.Mutation(pop, i, self.Mr, task, self.Rand)
			ind.evaluate(task, rnd=self.Rand)
			npop[i] = ind
			if npop[i].f < fxb: xb, fxb = self.get_best(npop[i], npop[i].f, xb, fxb)
		return npop, np.asarray([i.f for i in npop]), xb, fxb, args, dparams


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
