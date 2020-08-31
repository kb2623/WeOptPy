# encoding=utf8

"""Grey wolf optimizer module."""

import numpy as np

from WeOptPy.algorithms.interfaces.algorithm import Algorithm

__all__ = ['GreyWolfOptimizer']


class GreyWolfOptimizer(Algorithm):
	r"""Implementation of Grey wolf optimizer.

	Algorithm:
		Grey wolf optimizer

	Date:
		2018

	Author:
		Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		* Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
		* Grey Wold Optimizer (GWO) source code version 1.0 (MATLAB) from MathWorks

	Attributes:
		Name (List[str]): List of strings representing algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['GreyWolfOptimizer', 'GWO']

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

	def set_parameters(self, n=25, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			n (int): Number of individuals in population

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)

	def init_population(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. Additional arguments:
				4. Additional keyword arguments:
					* a (): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		pop, fpop, args, d = Algorithm.init_population(self, task)
		si = np.argsort(fpop)
		A, A_f, B, B_f, D, D_f = np.copy(pop[si[0]]), fpop[si[0]], np.copy(pop[si[1]]), fpop[si[1]], np.copy(pop[si[2]]), fpop[si[2]]
		d.update({'A': A, 'A_f': A_f, 'B': B, 'B_f': B_f, 'D': D, 'D_f': D_f})
		return pop, fpop, args, d

	def run_iteration(self, task, pop, fpop, xb, fxb, A, A_f, B, B_f, D, D_f, *args, **dparams):
		r"""Core function of GreyWolfOptimizer algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations function/fitness values.
			xb (numpy.ndarray):
			fxb (float):
			A (numpy.ndarray):
			A_f (float):
			B (numpy.ndarray):
			B_f (float):
			D (numpy.ndarray):
			D_f (float):
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population
				2. New population fitness/function values
				3. Additional arguments.
				4. Additional keyword arguments:
					* a (): TODO
		"""
		a = 2 - task.Evals * (2 / task.nFES)
		for i, w in enumerate(pop):
			A1, C1 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X1 = A - A1 * np.fabs(C1 * A - w)
			A2, C2 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X2 = B - A2 * np.fabs(C2 * B - w)
			A3, C3 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X3 = D - A3 * np.fabs(C3 * D - w)
			pop[i] = task.repair((X1 + X2 + X3) / 3, self.Rand)
			fpop[i] = task.eval(pop[i])
		for i, f in enumerate(fpop):
			if f < A_f: A, A_f = pop[i].copy(), f
			elif A_f < f < B_f: B, B_f = pop[i].copy(), f
			elif B_f < f < D_f: D, D_f = pop[i].copy(), f
		xb, fxb = self.get_best(A, A_f, xb, fxb)
		return pop, fpop, xb, fxb, args, {'A': A, 'A_f': A_f, 'B': B, 'B_f': B_f, 'D': D, 'D_f': D_f}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
