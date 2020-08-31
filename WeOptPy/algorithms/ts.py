# encoding=utf8

"""Tabu search algorithm module."""

# TODO implement algorithm

from numpy import random as rand

from WeOptPy.algorithms.interfaces import Algorithm

__all__ = ['TabuSearch']


class TabuSearch(Algorithm):
	r"""Implementation of Tabu Search Algorithm.

	Algorithm:
		Tabu Search Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://www.cleveralgorithms.com/nature-inspired/stochastic/tabu_search.html

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
	"""
	Name = ['TabuSearch', 'TS']

	@staticmethod
	def type_parameters():
		r"""Return functions for checking values of parameters.

		Return:
			Dict[str, Callable[[Any], bool]]:
				* n: Check if number of individuals is :math:`\in [0, \infty]`.
		"""
		return {
			'n': lambda x: isinstance(x, int) and x > 0
		}

	def set_parameters(self, **ukwargs):
		r"""Set the algorithm parameters/arguments."""
		Algorithm.set_parameters(self, **ukwargs)

	def move(self):
		r"""Move some."""
		return list()

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **kwargs):
		r"""Core function of the algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Individuals fitness/objective values.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solutions fitness/objective value.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
		"""
		return pop, fpop, xb, fxb, args, kwargs

	
# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
