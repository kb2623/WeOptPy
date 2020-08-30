# encoding=utf8

from WeOptPy.algorithms.de import cross_best1
from WeOptPy.algorithms.saba import SelfAdaptiveBatAlgorithm

__all__ = ['HybridSelfAdaptiveBatAlgorithm']


class HybridSelfAdaptiveBatAlgorithm(SelfAdaptiveBatAlgorithm):
	r"""Implementation of Hybrid self adaptive bat algorithm.

	Algorithm:
		Hybrid self adaptive bat algorithm

	Date:
		April 2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Fister, Iztok, Simon Fong, and Janez Brest. "a novel hybrid self-adaptive bat algorithm." The Scientific World Journal 2014 (2014).

	Reference URL:
		https://www.hindawi.com/journals/tswj/2014/709738/cta/

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		F (float): Scaling factor for local search.
		CR (float): Probability of crossover for local search.
		CrossMutt (Callable[[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, Dict[str, Any]): Local search method based of Differential evolution strategy.

	See Also:
		* :class:`NiaPy.algorithms.basic.BatAlgorithm`
	"""
	Name = ['HybridSelfAdaptiveBatAlgorithm', 'HSABA']

	@staticmethod
	def algorithm_info():
		r"""Get basic information about the algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Fister, Iztok, Simon Fong, and Janez Brest. "a novel hybrid self-adaptive bat algorithm." The Scientific World Journal 2014 (2014)."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.typeParameters`
		"""
		d = SelfAdaptiveBatAlgorithm.type_parameters()
		d.update({
			'F': lambda x: isinstance(x, (int, float)) and x > 0,
			'CR': lambda x: isinstance(x, float) and 0 <= x <= 1
		})
		return d

	def set_parameters(self, F=0.9, CR=0.85, CrossMutt=cross_best1, **ukwargs):
		r"""Set core parameters of HybridBatAlgorithm algorithm.

		Arguments:
			F (Optional[float]): Scaling factor for local search.
			CR (Optional[float]): Probability of crossover for local search.
			CrossMutt (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, Dict[str, Any], numpy.ndarray]]): Local search method based of Differential evolution strategy.
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.setParameters`
		"""
		SelfAdaptiveBatAlgorithm.set_parameters(self, **ukwargs)
		self.F, self.CR, self.CrossMutt = F, CR, CrossMutt

	def get_parameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]: Parameters of the algorithm.

		See Also:
			* :func:`NiaPy.algorithms.modified.AdaptiveBatAlgorithm.getParameters`
		"""
		d = SelfAdaptiveBatAlgorithm.get_parameters(self)
		d.update({
			'F': self.F,
			'CR': self.CR
		})
		return d

	def local_search(self, best, a, task, i, Sol, Fitness, **kwargs):
		r"""Improve the best solution.

		Args:
			best (numpy.ndarray): Global best individual.
			a (numpy.ndarray): TODO
			task (Task): Optimization task.
			i (int): Index of current individual.
			Sol (numpy.ndarray): Current population.
			Fitness (numpy.ndarray): Current populations fitness/objective values.
			kwargs (Dict[str, Any]):

		Returns:
			numpy.ndarray: New solution based on global best individual.
		"""
		return task.repair(self.CrossMutt(Sol, i, best, self.F, self.CR, rnd=self.Rand), rnd=self.Rand)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
