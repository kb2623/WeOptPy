# encoding=utf8

"""Moth flame optimizer algorithm module."""

import numpy as np

from WeOptPy.algorithms.interfaces.algorithm import Algorithm

__all__ = ['MothFlameOptimizer']


class MothFlameOptimizer(Algorithm):
	r"""MothFlameOptimizer of Moth flame optimizer.

	Algorithm:
		Moth flame optimizer

	Date:
		2018

	Author:
		Kivanc Guckiran and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Mirjalili, Seyedali. "Moth-flame optimization algorithm: a novel nature-inspired heuristic paradigm." Knowledge-Based Systems 89 (2015): 228-249.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['MothFlameOptimizer', 'MFO']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Mirjalili, Seyedali. "Moth-flame optimization algorithm: a novel nature-inspired heuristic paradigm." Knowledge-Based Systems 89 (2015): 228-249."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: TODO

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		return Algorithm.type_parameters()

	def set_parameters(self, n=25, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			n (int): Number of individuals in population

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)

	def init_population(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized population
				2. Initialized population function/fitness values
				3. Additional arguments:
				4. Additional keyword arguments:
					* best_flames (numpy.ndarray): Best individuals
					* best_flame_fitness (numpy.ndarray): Best individuals fitness/function values
					* previous_population (numpy.ndarray): Previous population
					* previous_fitness (numpy.ndarray[float]): Previous population fitness/function values

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		moth_pos, moth_fitness, args, d = Algorithm.init_population(self, task)
		# Create best population
		indexes = np.argsort(moth_fitness)
		best_flames, best_flame_fitness = moth_pos[indexes], moth_fitness[indexes]
		# Init previous population
		previous_population, previous_fitness = np.zeros((self.NP, task.D)), np.zeros(self.NP)
		d.update({'best_flames': best_flames, 'best_flame_fitness': best_flame_fitness, 'previous_population': previous_population, 'previous_fitness': previous_fitness})
		return moth_pos, moth_fitness, args, d

	def run_iteration(self, task, moth_pos, moth_fitness, xb, fxb, best_flames, best_flame_fitness, previous_population, previous_fitness, *args, **dparams):
		r"""Core function of MothFlameOptimizer algorithm.

		Args:
			task (Task): Optimization task.
			moth_pos (numpy.ndarray): Current population.
			moth_fitness (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Current population best individual.
			fxb (float): Current best individual.
			best_flames (numpy.ndarray): Best found individuals.
			best_flame_fitness (numpy.ndarray): Best found individuals fitness/function values.
			previous_population (numpy.ndarray): Previous population.
			revious_fitness (numpy.ndarray): Previous population fitness/function values.
			args (list): Additional parameters.
			dparams (dict): Additional parameters.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments.
				6. Additional keyword arguments:
					* best_flames (numpy.ndarray): Best individuals.
					* best_flame_fitness (numpy.ndarray): Best individuals fitness/function values.
					* previous_population (numpy.ndarray): Previous population.
					* previous_fitness (numpy.ndarray): Previous population fitness/function values.
		"""
		# Previous positions
		previous_population, previous_fitness = moth_pos, moth_fitness
		# Create sorted population
		indexes = np.argsort(moth_fitness)
		sorted_population = moth_pos[indexes]
		# Some parameters
		flame_no, a = round(self.NP - task.Iters * ((self.NP - 1) / task.nGEN)), -1 + task.Iters * ((-1) / task.nGEN)
		for i in range(self.NP):
			for j in range(task.D):
				distance_to_flame, b, t = abs(sorted_population[i, j] - moth_pos[i, j]), 1, (a - 1) * self.rand() + 1
				if i <= flame_no: moth_pos[i, j] = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + sorted_population[i, j]
				else: moth_pos[i, j] = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + sorted_population[flame_no, j]
		moth_pos = np.apply_along_axis(task.repair, 1, moth_pos, self.Rand)
		moth_fitness = np.apply_along_axis(task.eval, 1, moth_pos)
		xb, fxb = self.get_best(moth_pos, moth_fitness, xb, fxb)
		double_population, double_fitness = np.concatenate((previous_population, best_flames), axis=0), np.concatenate((previous_fitness, best_flame_fitness), axis=0)
		indexes = np.argsort(double_fitness)
		double_sorted_fitness, double_sorted_population = double_fitness[indexes], double_population[indexes]
		for newIdx in range(2 * self.NP): double_sorted_population[newIdx] = np.array(double_population[indexes[newIdx], :])
		best_flame_fitness, best_flames = double_sorted_fitness[:self.NP], double_sorted_population[:self.NP]
		return moth_pos, moth_fitness, xb, fxb, args, {'best_flames': best_flames, 'best_flame_fitness': best_flame_fitness, 'previous_population': previous_population, 'previous_fitness': previous_fitness}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
