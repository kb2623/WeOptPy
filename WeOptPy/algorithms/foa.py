# encoding=utf8

"""Forest optimization algorithm module."""

import numpy as np

from WeOptPy.util.utility import limit_repair
from WeOptPy.algorithms.interfaces.algorithm import Algorithm

__all__ = ['ForestOptimizationAlgorithm']


class ForestOptimizationAlgorithm(Algorithm):
	r"""Implementation of Forest Optimization Algorithm.

	Algorithm:
		Forest Optimization Algorithm

	Date:
		2019

	Authors:
		Luka Pečnik and Klemen Berkovic

	License:
		MIT

	Reference paper:
		* Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174,
		* https://doi.org/10.1016/j.eswa.2014.05.009.

	References URL:
		Implementation is based on the following MATLAB code: https://github.com/cominsys/FOA

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		lt (int): Life time of trees parameter.
		al (int): Area limit parameter.
		lsc (int): Local seeding changes parameter.
		gsc (int): Global seeding changes parameter.
		tr (float): Transfer rate parameter.

	See Also:
		* :class:`WeOptPy.algorithms.Algorithm`
	"""
	Name = ['ForestOptimizationAlgorithm', 'FOA']

	@staticmethod
	def algorithm_info():
		r"""Get algorithms information.

		Returns:
			str: Algorithm information.
		"""
		return r"""
			Description: Forest Optimization Algorithm is inspired by few trees in the forests which can survive for several decades, while other trees could live for a limited period.
			Authors: Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi
			Year: 2014
			Main reference: Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.
		"""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* lt (Callable[[int], bool]): Checks if life time parameter has a proper value.
				* al (Callable[[int], bool]): Checks if area limit parameter has a proper value.
				* lsc (Callable[[int], bool]): Checks if local seeding changes parameter has a proper value.
				* gsc (Callable[[int], bool]): Checks if global seeding changes parameter has a proper value.
				* tr (Callable[[float], bool]): Checks if transfer rate parameter has a proper value.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'lt': lambda x: isinstance(x, int) and x > 0,
			'al': lambda x: isinstance(x, int) and x > 0,
			'lsc': lambda x: isinstance(x, int) and x > 0,
			'gsc': lambda x: isinstance(x, int) and x > 0,
			'tr': lambda x: isinstance(x, float) and 0 <= x <= 1,
		})
		return d

	def set_parameters(self, n=10, lt=3, al=10, lsc=1, gsc=1, tr=0.3, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			n (Optional[int]): Population size.
			lt (Optional[int]): Life time parameter.
			al (Optional[int]): Area limit parameter.
			lsc (Optional[int]): Local seeding changes parameter.
			gsc (Optional[int]): Global seeding changes parameter.
			tr (Optional[float]): Transfer rate parameter.
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)
		self.lt, self.al, self.lsc, self.gsc, self.tr = lt, al, lsc, gsc, tr

	def get_parameters(self):
		r"""Get parameters values of the algorithm.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'lt': self.lt,
			'al': self.al,
			'lsc': self.lsc,
			'gsc': self.gsc,
			'tr': self.tr
		})
		return d

	def local_seeding(self, task, trees):
		r"""Local optimum search stage.

		Args:
			task (Task): Optimization task.
			trees (numpy.ndarray): Zero age trees for local seeding.

		Returns:
			numpy.ndarray: Resulting zero age trees.
		"""
		new_trees = []
		for tree in trees:
			new_tree = tree.copy()
			i_move = self.randint(task.D, self.lsc)
			new_tree[i_move] = self.uniform(-self.dx[i_move], self.dx[i_move], len(i_move) if isinstance(i_move, np.ndarray) else 1)
			new_trees.append(limit_repair(new_tree, task.Lower, task.Upper))
		return np.asarray(new_trees)

	def global_seeding(self, task, no_candidate):
		r"""Global optimum search stage that should prevent getting stuck in a local optimum.

		Args:
			task (Task): Optimization task.
			no_candidate (int): Number of new candidates.

		Returns:
			numpy.ndarray: Resulting trees.
		"""
		return self.uniform(task.Lower, task.Upper, (no_candidate, task.D))

	def remove_life_time_exceeded(self, trees, candidates, age):
		r"""Remove dead trees.

		Args:
			trees (numpy.ndarray): Population to test.
			candidates (numpy.ndarray): Candidate population array to be updated.
			age (numpy.ndarray): Age of trees.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
				1. Alive trees.
				2. New candidate population.
				3. Age of trees.
		"""
		lifeTimeExceeded = np.where(age > self.lt)
		candidates = trees[lifeTimeExceeded]
		trees = np.delete(trees, lifeTimeExceeded, axis=0)
		age = np.delete(age, lifeTimeExceeded, axis=0)
		return trees, candidates, age

	def survival_of_the_fittest(self, task, trees, candidates, age):
		r"""Evaluate and filter current population.

		Args:
			task (Task): Optimization task.
			trees (numpy.ndarray): Population to evaluate.
			candidates (numpy.ndarray): Candidate population array to be updated.
			age (numpy.ndarray): Age of trees.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
				1. Trees sorted by fitness value.
				2. Updated candidate population.
				3. Population fitness values.
				4. Age of trees
		"""
		evaluations = np.apply_along_axis(task.eval, 1, trees)
		ei = evaluations.argsort()
		candidates = np.append(candidates, trees[ei[self.al:]], axis=0)
		trees = trees[ei[:self.al]]
		age = age[ei[:self.al]]
		evaluations = evaluations[ei[:self.al]]
		return trees, candidates, evaluations, age

	def init_population(self, task):
		r"""Initialize the starting population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* age (numpy.ndarray): Age of trees.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.initPopulation`
		"""
		Trees, Evaluations, args, _ = Algorithm.init_population(self, task)
		age = np.zeros(self.NP, dtype=np.int32)
		self.dx = np.absolute(task.Upper) / 5
		return Trees, Evaluations, args, {'age': age}

	def run_iteration(self, task, Trees, Evaluations, xb, fxb, age, *args, **kwargs):
		r"""Core function of Forest Optimization Algorithm.

		Args:
			task (Task): Optimization task.
			Trees (numpy.ndarray): Current population.
			Evaluations (numpy.ndarray): Current population function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individual fitness/function value.
			age (numpy.ndarray): Age of trees.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. Global best individual.
				4. Global best individual's fitness value.
				5. Additional arguments.
				6. Additional keyword arguments:
					* age (numpy.ndarray): Age of trees.
		"""
		candidatePopulation = np.ndarray((0, task.D + 1))
		zeroAgeTrees = Trees[age == 0]
		localSeeds = self.local_seeding(task, zeroAgeTrees)
		age += 1
		Trees, candidatePopulation, age = self.remove_life_time_exceeded(Trees, candidatePopulation, age)
		Trees = np.append(Trees, localSeeds, axis=0)
		age = np.append(age, np.zeros(len(localSeeds), dtype=np.int32))
		Trees, candidatePopulation, Evaluations, age = self.survival_of_the_fittest(task, Trees, candidatePopulation, age)
		gsn = int(self.tr * len(candidatePopulation))
		if gsn > 0:
			globalSeeds = self.global_seeding(task, gsn)
			Trees = np.append(Trees, globalSeeds, axis=0)
			age = np.append(age, np.zeros(len(globalSeeds), dtype=np.int32))
			gste = np.apply_along_axis(task.eval, 1, globalSeeds)
			Evaluations = np.append(Evaluations, gste)
		ib = np.argmin(Evaluations)
		age[ib] = 0
		if Evaluations[ib] < fxb: xb, fxb = Trees[ib].copy(), Evaluations[ib]
		return Trees, Evaluations, xb, fxb, args, {'age': age}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
