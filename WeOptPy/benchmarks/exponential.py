# encoding=utf8

"""Implementation of Exponential benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import exponential_function

__all__ = ["Exponential"]


class Exponential(Benchmark):
	r"""Implementations of Exponential functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Exponential Function

		:math:`f(\mathbf{x}) = -e^{-0.5 \sum_{i=1}^N x_i^2}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = -1`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = -e^{-0.5 \sum_{i=1}^N x_i^2}$

		Equation:
			\begin{equation}f(\mathbf{x}) = -e^{-0.5 \sum_{i=1}^N x_i^2} \end{equation}

		Domain:
			$-1 \leq x_i \leq 1$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.Easom

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Exponential"]

	def __init__(self, Lower=-1.0, Upper=1.0, **kwargs):
		r"""Initialize Exponential benchmark.

		Args:
			Lower (Union[int, float, numpy.ndarray]): Lower bound of problem.
			Upper (Union[int, float, numpy.ndarray]): Upper bound of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.dsa

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper, **kwargs)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\mathbf{x}) = -e^{-0.5 \sum_{i=1}^N x_i^2}$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: exponential_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
