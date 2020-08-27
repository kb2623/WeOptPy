# encoding=utf8
import copy

import numpy as np

from WeOptPy.algorithms.interfaces import (
	Algorithm,
	Individual,
	default_individual_init
)

__all__ = ['ArtificialBeeColonyAlgorithm']


class SolutionABC(Individual):
	r"""Representation of solution for Artificial Bee Colony Algorithm.

	Date:
		2018

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, **kargs):
		r"""Initialize individual.

		Args:
			kargs (dict): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Individual.__init__`
		"""
		Individual.__init__(self, **kargs)


class ArtificialBeeColonyAlgorithm(Algorithm):
	r"""Implementation of Artificial Bee Colony algorithm.

	Algorithm:
		Artificial Bee Colony algorithm

	Date:
		2018

	Author:
		Uroš Mlakar and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Karaboga, d., and Bahriye B. "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm." Journal of global optimization 39.3 (2007): 459-471.

	Arguments
		Name (List[str]): List containing strings that represent algorithm names.
		Limit (Union[float, numpy.ndarray[float]]): Limit.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['ArtificialBeeColonyAlgorithm', 'ABC']

	@staticmethod
	def type_parameters():
		r"""Return functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* Limit (Callable[Union[float, numpy.ndarray[float]]]): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({'Limit': lambda x: isinstance(x, int) and x > 0})
		return d

	def set_parameters(self, n=10, limit=100, **ukwargs):
		r"""Set the parameters of Artificial Bee Colony Algorithm.

		Args:
			n (int): Number of individuals in population.
			limit (Optional[Union[float, numpy.ndarray[float]]]): Limit.
			ukwargs (dict): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, init_pop_func=default_individual_init, itype=SolutionABC, **ukwargs)
		self.FoodNumber, self.Limit = int(self.NP / 2), limit

	def calculate_probs(self, foods):
		r"""Calculate the probes.

		Args:
			foods (numpy.ndarray): TODO

		Returns:
			numpy.ndarray: TODO
		"""
		probs = [1.0 / (foods[i].f + 0.01) for i in range(self.FoodNumber)]
		s = np.sum(probs)
		probs = [probs[i] / s for i in range(self.FoodNumber)]
		return probs

	def init_population(self, task):
		r"""Initialize the starting population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population
				2. New population fitness/function values
				3. Additional arguments:
					* Probes (numpy.ndarray): TODO
					* Trial (numpy.ndarray): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		foods, fpop, _ = Algorithm.init_population(self, task)
		probs, trial = np.full(self.FoodNumber, 0.0), np.full(self.FoodNumber, 0.0)
		return foods, fpop, {'probs': probs, 'trial': trial}

	def run_iteration(self, task, foods, fpop, xb, fxb, probs, trial, **dparams):
		r"""Core function of  the algorithm.

		Parameters:
			task (Task): Optimization task
			foods (numpy.ndarray): Current population
			fpop (numpy.ndarray[float]): Function/fitness values of current population
			xb (numpy.ndarray): Current best individual
			fxb (float): Current best individual fitness/function value
			probs (numpy.ndarray): TODO
			trial (numpy.ndarray): TODO
			dparams (Dict[str, Any]): Additional parameters

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population
				2. New population fitness/function values
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments:
					* Probes (numpy.ndarray): TODO
					* Trial (numpy.ndarray): TODO
		"""
		for i in range(self.FoodNumber):
			new_solution = copy.deepcopy(foods[i])
			param2change = int(self.rand() * task.D)
			neighbor = int(self.FoodNumber * self.rand())
			new_solution.x[param2change] = foods[i].x[param2change] + (-1 + 2 * self.rand()) * (foods[i].x[param2change] - foods[neighbor].x[param2change])
			new_solution.evaluate(task, rnd=self.Rand)
			if new_solution.f < foods[i].f:
				foods[i], trial[i] = new_solution, 0
				if new_solution.f < fxb: xb, fxb = new_solution.x.copy(), new_solution.f
			else: trial[i] += 1
		probs, t, s = self.calculate_probs(foods), 0, 0
		while t < self.FoodNumber:
			if self.rand() < probs[s]:
				t += 1
				solution = copy.deepcopy(foods[s])
				param2change = int(self.rand() * task.D)
				neighbor = int(self.FoodNumber * self.rand())
				while neighbor == s: neighbor = int(self.FoodNumber * self.rand())
				solution.x[param2change] = foods[s].x[param2change] + (-1 + 2 * self.rand()) * (foods[s].x[param2change] - foods[neighbor].x[param2change])
				solution.evaluate(task, rnd=self.Rand)
				if solution.f < foods[s].f:
					foods[s], trial[s] = solution, 0
					if solution.f < fxb: xb, fxb = solution.x.copy(), solution.f
				else: trial[s] += 1
			s += 1
			if s == self.FoodNumber: s = 0
		mi = np.argmax(trial)
		if trial[mi] >= self.Limit:
			foods[mi], trial[mi] = SolutionABC(task=task, rnd=self.Rand), 0
			if foods[mi].f < fxb: xb, fxb = foods[mi].x.copy(), foods[mi].f
		return foods, np.asarray([f.f for f in foods]), xb, fxb, {'probs': probs, 'trial': trial}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
