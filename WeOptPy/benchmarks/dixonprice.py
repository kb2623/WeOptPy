# encoding=utf8

"""Implementation of Dixon Price benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import dixon_price_function

__all__ = ["DixonPrice"]


class DixonPrice(Benchmark):
	r"""Implementations of Dixon Price function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Levy Function

		:math:`f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (2^{-\frac{2^1 - 2}{2^1}}, \cdots , 2^{-\frac{2^i - 2}{2^i}} , \cdots , 2^{-\frac{2^D - 2}{2^D}})`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2$

		Equation:
			\begin{equation} f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2 \end{equation}

		Domain:
			:math:`-10 \leq x_i \leq 10`

	Reference:
		https://www.sfu.ca/~ssurjano/dixonpr.html
	"""
	Name = ["DixonPrice"]

	def __init__(self, Lower=-10.0, Upper=10, **kwargs):
		r"""Initialize of Dixon Price benchmark.

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
		return r"""$f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float] Evaluation function.
		"""
		return lambda sol, **a: dixon_price_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
