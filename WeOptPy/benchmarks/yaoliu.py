# encoding=utf8

"""Implementation of YaoLiu benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import yaoliu_09_function

__all__ = ["YaoLiu09"]


class YaoLiu09(Benchmark):
	r"""Implementations of YaoLin09 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		YaoLiu09 Function

		:math:`f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10 \end{equation}

		Domain:
			$-5.12 \leq x_i \leq 5.12$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_Y.html#go_benchmark.YaoLiu09
	"""
	Name = ["YaoLiu09"]

	def __init__(self, Lower=-5.12, Upper=5.12, **kwargs):
		r"""Initialize YaoLiu09 benchmark.

		Args:
			Lower (Union[int, float, numpy.ndarray]): Lower bound of problem.
			Upper (Union[int, float, numpy.ndarray]): Upper bound of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: yaoliu_09_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
