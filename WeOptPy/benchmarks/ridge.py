# encoding=utf8

"""Implementation of Ridge benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import ridge_function

__all__ = ["Ridge"]

class Ridge(Benchmark):
	r"""Implementation of Ridge function.

	Date:
		2018

	Author:
		Klemen Brekovič

	License:
		MIT

	Function:
		Ridge function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-64, 64]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 \end{equation}

		Domain:
			$-64 \leq x_i \leq 64$

	Reference:
		http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Ridge"]

	def __init__(self, Lower=-64.0, Upper=64.0, **kwargs):
		"""Initialize Ridge benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: ridge_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
