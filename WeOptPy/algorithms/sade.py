# encoding=utf8

"""Adaptive differential evolution module."""

from WeOptPy.algorithms.de import (
	# CrossBest1,
	# CrossRand1,
	# CrossCurr2Best1,
	# CrossBest2,
	# CrossCurr2Rand1,
	# proportional,
	DifferentialEvolution
)

__all__ = [
	'StrategyAdaptationDifferentialEvolution',
	'StrategyAdaptationDifferentialEvolutionV1'
]


class StrategyAdaptationDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorihtm.

	Algorithm:
		Differential Evolution Algorithm With StrategyAdaptation

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/1554904

	Reference paper:
		Qin, a. Kai, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm for numerical optimization." 2005 IEEE congress on evolutionary computation. Vol. 2. IEEE, 2005.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`WeOptPy.algorithms.DifferentialEvolution`
	"""
	Name = ['StrategyAdaptationDifferentialEvolution', 'SADE', 'SaDE']

	@staticmethod
	def algorithm_info():
		r"""Geg basic algorithm information.

		Returns:
			str: Basic algorithm information.

		See Also:
			* :func:`NiaPy.algorithms.interfaces.Algorithm.algorithm_info`
		"""
		return r"""Qin, a. Kai, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm for numerical optimization." 2005 IEEE congress on evolutionary computation. Vol. 2. IEEE, 2005."""

	def set_parameters(self, **kwargs):
		r"""Set the algorithm parameters.

		Args:
			kwargs (dict): Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.interfaces.Algorithm.set_parameters`
		"""
		DifferentialEvolution.set_parameters(self, **kwargs)
		# TODO add parameters of the algorithm

	def get_parameters(self):
		r"""Get algorithm parameter values.

		Returns:
			Dict[str, Any]: TODO
		"""
		d = DifferentialEvolution.get_parameters(self)
		# TODO add paramters values
		return d

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **kwargs):
		r"""Core function of the algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current population's fitness values.
			xb (numpy.ndarray): Current global best individual.
			fxb (float): Current global best individual's best fitness value.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.DifferentialEvolution.evolve`
			* :func:`WeOptPy.algorithms.DifferentialEvolution.selection`
			* :func:`WeOptPy.algorithms.DifferentialEvolution.post_selection`
		"""
		# TODO implemnt algorithm
		return pop, fpop, xb, fxb, args, kwargs


class StrategyAdaptationDifferentialEvolutionV1(DifferentialEvolution):
	r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorithm.

	Algorithm:
		Differential Evolution Algorithm With StrategyAdaptation

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/4632146

	Reference paper:
		Qin, a. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential evolution algorithm with strategy adaptation for global numerical optimization." IEEE transactions on Evolutionary Computation 13.2 (2009): 398-417.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['StrategyAdaptationDifferentialEvolutionV1', 'SADEV1', 'SaDEV1']

	@staticmethod
	def algorithm_info():
		r"""Get algorithm information.

		Returns:
			str: Get algorithm information.

		See Also:
			* :func:`WeOptPy.algorithms.interfaces.Algorithm.algorithm_info`
		"""
		return r"""Qin, a. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential evolution algorithm with strategy adaptation for global numerical optimization." IEEE transactions on Evolutionary Computation 13.2 (2009): 398-417."""

	def set_parameters(self, **kwargs):
		r"""Set algorithm parameters.

		Args:
			**kwargs (dict): Additional keyword arguments.
		"""
		DifferentialEvolution.set_parameters(self, **kwargs)
		# TODO add parameters of the algorithm

	def get_parameters(self):
		r"""Get parameter values of the algorithm.

		Returns:
			Dict[str, Any]: TODO
		"""
		d = DifferentialEvolution.get_parameters(self)
		# TODO add parameters values
		return d

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **kwargs):
		r"""Core function of Differential Evolution algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Current best individual.
			fxb (float): Current best individual function/fitness value.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.DifferentialEvolution.evolve`
			* :func:`WeOptPy.algorithms.DifferentialEvolution.selection`
			* :func:`WeOptPy.algorithms.DifferentialEvolution.postSelection`
		"""
		# TODO implement algorithm
		return pop, fpop, xb, fxb, args, kwargs
