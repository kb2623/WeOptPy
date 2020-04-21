# encoding=utf8

"""Implementation of Infinity benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import infinity_function

__all__ = ["Infinity"]


class Infinity(Benchmark):
	r"""Implementations of Infinity function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Infinity Function

		:math:`f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right) \end{equation}

		Domain:
			:math:`-1 \leq x_i \leq 1`

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_I.html#go_benchmark.Infinity
	"""
	Name: List[str] = ["Infinity"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -1.0, Upper: Union[int, float, np.ndarray] = 1.0, **kwargs):
		r"""Initialize Infinity benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x: infinity_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
