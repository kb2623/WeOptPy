# encoding=utf8

import numpy as np

from WeOptPy.algorithms.interfaces import Algorithm

__all__ = ['SineCosineAlgorithm']


class SineCosineAlgorithm(Algorithm):
	r"""Implementation of sine cosine algorithm.

	Algorithm:
		Sine Cosine Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705115005043

	Reference paper:
		Seyedali Mirjalili, SCA: a Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022.

	Attributes:
		Name (List[str]): List of string representing algorithm names.
		a (float): Parameter for control in :math:`r_1` value
		Rmin (float): Minimu value for :math:`r_3` value
		Rmax (float): Maximum value for :math:`r_3` value

	See Also:
		* :class:`WeOptPy.algorithms.Algorithm`
	"""
	Name = ['SineCosineAlgorithm', 'SCA']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Seyedali Mirjalili, SCA: a Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* a (Callable[[Union[float, int]], bool]): TODO.
				* Rmin (Callable[[Union[float, int]], bool]): TODO.
				* Rmax (Callable[[Union[float, int]], bool]): TODO.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'a': lambda x: isinstance(x, (float, int)) and x > 0,
			'Rmin': lambda x: isinstance(x, (float, int)),
			'Rmax': lambda x: isinstance(x, (float, int))
		})
		return d

	def set_parameters(self, n=25, a=3, Rmin=0, Rmax=2, **ukwargs):
		r"""Set the arguments of an algorithm.

		Args:
			n (Optional[int]): Number of individual in population
			a (Optional[float]): Parameter for control in :math:`r_1` value
			Rmin (Optional[float]): Minimu value for :math:`r_3` value
			Rmax (Optional[float]): Maximum value for :math:`r_3` value

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)
		self.a, self.Rmin, self.Rmax = a, Rmin, Rmax

	def get_parameters(self):
		r"""Get algorithm parameters values.

		Returns:
			Dict[str, Any]: TODO.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.getParameters`
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'a': self.a,
			'Rmin': self.Rmin,
			'Rmax': self.Rmax
		})
		return d

	def next_pos(self, x, x_b, r1, r2, r3, r4, task):
		r"""Move individual to new position in search space.

		Args:
			x (numpy.ndarray): Individual represented with components.
			x_b (nmppy.ndarray): Best individual represented with components.
			r1 (float): Number dependent on algorithm iteration/generations.
			r2 (float): Random number in range of 0 and 2 * PI.
			r3 (float): Random number in range [Rmin, Rmax].
			r4 (float): Random number in range [0, 1].
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: New individual that is moved based on individual ``x``.
		"""
		return task.repair(x + r1 * (np.sin(r2) if r4 < 0.5 else np.cos(r2)) * np.fabs(r3 * x_b - x), self.Rand)

	def init_population(self, task):
		r"""Initialize the individuals.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized population of individuals
				2. Function/fitness values for individuals
				3. Additional arguments
				4. Additional keyword arguments
		"""
		return Algorithm.init_population(self, task)

	def run_iteration(self, task, P, P_f, xb, fxb, *args, **dparams):
		r"""Core function of Sine Cosine Algorithm.

		Args:
			task (Task): Optimization task.
			P (numpy.ndarray): Current population individuals.
			P_f (numpy.ndarray[float]): Current population individulas function/fitness values.
			xb (numpy.ndarray): Current best solution to optimization task.
			fxb (float): Current best function/fitness value.
			args (list): Additional parameters.
			dparams (dict): Additional keyword parameters.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments.
				6. Additional keyword arguments.
		"""
		r1, r2, r3, r4 = self.a - task.Iters * (self.a / task.Iters), self.uniform(0, 2 * np.pi), self.uniform(self.Rmin, self.Rmax), self.rand()
		P = np.apply_along_axis(self.next_pos, 1, P, xb, r1, r2, r3, r4, task)
		P_f = np.apply_along_axis(task.eval, 1, P)
		xb, fxb = self.get_best(P, P_f, xb, fxb)
		return P, P_f, xb, fxb, args, dparams

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
