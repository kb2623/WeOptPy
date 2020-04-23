# encoding=utf8

"""Implementations of Zakharov function."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import zakharov_function

__all__ = ["Zakharov"]


class Zakharov(Benchmark):
	r"""Implementations of Zakharov functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Levy Function

		:math:`f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4 \end{equation}

	Domain:
		:math:`-5 \leq x_i \leq 10`

	Reference:
		https://www.sfu.ca/~ssurjano/levy.html

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Zakharov"]

	def __init__(self, Lower=-5.0, Upper=10.0, **kwargs):
		r"""Initialize Zakharov benchmark.

		Args:
			Lower (Union[int, float, numpy.ndarray]): Lower bound of problem.
			Upper (Union[int, float, numpy.ndarray]): Upper bound of problem.

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
		return r"""$f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda sol, **a: zakharov_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
