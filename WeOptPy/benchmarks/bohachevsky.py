# encoding=utf8

"""Implementation of HGBat benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import bohachevsky_function

__all__ = ["Bohachevsky"]


class Bohachevsky(Benchmark):
	r"""Implementations of Bohachevsky functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Bohachevsky Function

		:math:`f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-15, 15]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7$

		Equation:
			\begin{equation}  f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7 \end{equation}

		Domain:
			$-15 \leq x_i \leq 15$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bohachevsky

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Bohachevsky"]

	def __init__(self, Lower=-15.0, Upper=15.0, **kwargs):
		r"""Initialize Bohachevsky benchmark.

		Args:
			Lower (Optional[Union[int, float, numpy.ndarray]]): Lower bound of problem.
			Upper (Optional[Union[int, float, numpy.ndarray]]): Upper bound of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper, **kwargs)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: latex code.
		"""
		return r"""f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: bohachevsky_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
