# encoding=utf8

"""Implementation of benchmarks utility function."""

import numpy as np


__all__ = ['UtilityFunction']


class UtilityFunction:
	r"""Class representing benchmarks.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of names representing benchmark names.
		lower (Union[int, float, list, numpy.ndarray]): lower bounds.
		upper (Union[int, float, list, numpy.ndarray]): upper bounds.
	"""
	Name = ['UtilityFunction']

	def __init__(self, lower, upper, **kwargs):
		r"""Initialize benchmark.

		Args:
			lower (Union[int, float, list, numpy.ndarray]): lower bounds.
			upper (Union[int, float, list, numpy.ndarray]): upper bounds.
			kwargs (Dict[str, Any]): Additional arguments.
		"""
		self.Lower, self.Upper = lower, upper

	@staticmethod
	def latex_code():
		r"""Return the latex code of the problem.

		Returns:
			str: Latex code
		"""
		return r'''$f(x) = \infty$'''

	def function(self):
		r"""Get the optimization function.

		Returns:
			Callable[[Union[list, numpy.ndarray], Dict[str, Any]], float]: Fitness function.
		"""
		def fun(x, **kwargs):
			r"""Initialize benchmark.

			Args:
				x (Union[int, float, list, numpy.ndarray]): Solution to the problem.
				kwargs (Dict[str, Any]): Additional arguments for the objective/utility/fitness function.

			Returns:
				float: Fitness value for the solution
			"""
			return np.sum(x ** 2)
		return fun

	def __call__(self):
		r"""Get the optimization function.

		Returns:
			Callable[[Union[list, numpy.ndarray], Dict[str, Any]], float]: Fitness funciton.
		"""
		return self.function()


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
