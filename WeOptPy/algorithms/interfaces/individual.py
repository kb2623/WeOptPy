# encoding=utf8

"""Individual interface module."""

import numpy as np
from numpy import random as rand

from WeOptPy.util.utility import objects2array

__all__ = [
	'Individual',
	'default_numpy_init',
	'default_individual_init',
]


class Individual:
	r"""Class that represents one solution in population of solutions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		x (numpy.ndarray): Coordinates of individual.
		f (float): Function/fitness value of individual.
	"""
	x = None
	f = np.inf

	def __init__(self, x=None, task=None, e=True, rnd=rand, **kwargs):
		r"""Initialize new individual.

		Parameters:
			x (Optional[numpy.ndarray]): Individuals components.
			task (Optional[Task]): Optimization task.
			e (Optional[bool]): True to evaluate the individual on initialization. Default value is True.
			rand (Optional[rand.RandomState]): Random generator.
			kwargs (Dict[str, Any]): Additional arguments.
		"""
		self.f = task.optType.value * np.inf if task is not None else np.inf
		if x is not None: self.x = x if isinstance(x, np.ndarray) else np.asarray(x)
		else: self.generate_solution(task, rnd)
		if e and task is not None: self.evaluate(task, rnd)

	def generate_solution(self, task, rnd=rand):
		r"""Generate new solution.

		Generate new solution for this individual and set it to ``self.x``.
		This method uses ``rnd`` for getting random numbers.
		For generating random components ``rnd`` and ``task`` is used.

		Args:
			task (Task): Optimization task.
			rnd (Optional[rand.RandomState]): Random numbers generator object.
		"""
		if task is not None: self.x = task.Lower + task.bRange * rnd.rand(task.D)

	def evaluate(self, task, rnd=rand):
		r"""Evaluate the solution.

		Evaluate solution ``this.x`` with the help of task.
		Task is used for repairing the solution and then evaluating it.

		Args:
			task (Task): Objective function object.
			rnd (Optional[rand.RandomState]): Random generator.

		See Also:
			* :func:`NiaPy.util.Task.repair`
		"""
		self.x = task.repair(self.x, rnd=rnd)
		self.f = task.eval(self.x)

	def copy(self):
		r"""Return a copy of self.

		Method returns copy of ``this`` object so it is safe for editing.

		Returns:
			Individual: Copy of self.
		"""
		return Individual(x=self.x.copy(), f=self.f, e=False)

	def __eq__(self, other):
		r"""Compare the individuals for equalities.

		Args:
			other (Union[Any, n.ndarray]): Object that we want to compare this object to.

		Returns:
			bool: `True` if equal or `False` if no equal.
		"""
		if isinstance(other, np.ndarray):
			for e in other:
				if self == e: return True
			return False
		return np.array_equal(self.x, other.x) and self.f == other.f

	def __str__(self):
		r"""Print the individual with the solution and objective value.

		Returns:
			str: String representation of self.
		"""
		return '%s -> %s' % (self.x, self.f)

	def __getitem__(self, i):
		r"""Get the value of i-th component of the solution.

		Args:
			i (int): Position of the solution component.

		Returns:
			Any: Value of ith component.
		"""
		return self.x[i]

	def __setitem__(self, i, v):
		r"""Set the value of i-th component of the solution to v value.

		Args:
			i (int): Position of the solution component.
			v (Any): Value to set to i-th component.
		"""
		self.x[i] = v

	def __len__(self):
		r"""Get the length of the solution or the number of components.

		Returns:
			int: Number of components.
		"""
		return len(self.x)


def default_numpy_init(task, n, rnd=rand, *args, **kwargs):
	r"""Initialize starting population that is represented with `numpy.ndarray` with shape `{n, task.d}`.

	Args:
		task (Task): Optimization task.
		n (int): Number of individuals in population.
		rnd (Optional[rand.RandomState]): Random number generator.
		args (list): Additional arguments.
		kwargs (dict): Additional Keyword arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray]:
			1. New population with shape `{n, task.d}`.
			2. New population function/fitness values.
	"""
	pop = task.Lower + rnd.rand(n, task.D) * task.bRange
	fpop = np.apply_along_axis(task.eval, 1, pop)
	return pop, fpop, args, kwargs


def default_individual_init(task, n, rnd=rand, itype=None, *args, **kwargs):
	r"""Initialize `n` individuals of type `itype`.

	Args:
		task (Task): Optimization task.
		n (int): Number of individuals in population.
		rnd (Optional[rand.RandomState]): Random number generator.
		itype (Optional[Individual]): Class of individual in population.
		args (list): Additional arguments.
		kwargs (dict): Additional keyword arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray]:
			1. Initialized individuals.
			2. Initialized individuals function/fitness values.
			3. Additional arguments.
			4. Additional keyword arguments.
	"""
	pop = objects2array([itype(task=task, rnd=rnd, e=True) for _ in range(n)])
	return pop, np.asarray([x.f for x in pop]), args, kwargs


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
