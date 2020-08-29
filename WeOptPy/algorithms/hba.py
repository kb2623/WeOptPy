# encoding=utf8

from WeOptPy.algorithms.ba import BatAlgorithm
from WeOptPy.algorithms.de import CrossBest1

__all__ = ['HybridBatAlgorithm']


class HybridBatAlgorithm(BatAlgorithm):
	r"""Implementation of Hybrid bat algorithm.

	Algorithm:
		Hybrid bat algorithm

	Date:
		2018

	Author:
		Grega Vrbancic and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "a Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		F (float): Scaling factor.
		CR (float): Crossover.

	See Also:
		* :class:`NiaPy.algorithms.basic.BatAlgorithm`
	"""
	Name = ['HybridBatAlgorithm', 'HBA']

	@staticmethod
	def algorithm_info():
		r"""Get basic information about the algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "a Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* F (Callable[[Union[int, float]], bool]): Scaling factor.
				* CR (Callable[[float], bool]): Crossover probability.

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.typeParameters`
		"""
		d = BatAlgorithm.type_parameters()
		d.update({
			'F': lambda x: isinstance(x, (int, float)) and x > 0,
			'CR': lambda x: isinstance(x, float) and 0 <= x <= 1
		})
		return d

	def set_parameters(self, F=0.50, CR=0.90, CrossMutt=CrossBest1, **ukwargs):
		r"""Set core parameters of HybridBatAlgorithm algorithm.

		Arguments:
			F (Optional[float]): Scaling factor.
			CR (Optional[float]): Crossover.

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.setParameters`
		"""
		BatAlgorithm.set_parameters(self, **ukwargs)
		self.F, self.CR, self.CrossMutt = F, CR, CrossMutt

	def local_search(self, best, task, i, Sol, **kwargs):
		r"""Improve the best solution.

		Args:
			best (numpy.ndarray): Global best individual.
			task (Task): Optimization task.
			i (int): Index of current individual.
			Sol (numpy.ndarray): Current best population.
			kwargs (Dict[str, Any]):

		Returns:
			numpy.ndarray: New solution based on global best individual.
		"""
		return task.repair(self.CrossMutt(Sol, i, best, self.F, self.CR, rnd=self.Rand), rnd=self.Rand)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
