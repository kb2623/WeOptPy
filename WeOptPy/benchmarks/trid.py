# encoding=utf8
"""Implementations of Levy function."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import trid_function

__all__ = ["Trid"]


class Trid(Benchmark):
	r"""Implementations of Trid functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Levy Function

		:math:`f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-D^2, D^2]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(\textbf{x}^*) = \frac{-D(D + 4)(D - 1)}{6}` at :math:`\textbf{x}^* = (1 (D + 1 - 1), \cdots , i (D + 1 - i) , \cdots , D (D + 1 - D))`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1} \end{equation}

		Domain:
			$-D^2 \leq x_i \leq D^2$

	Reference:
		https://www.sfu.ca/~ssurjano/trid.html

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Trid"]

	def __init__(self, D=2, **kwargs):
		r"""Initialize Trid benchmark.

		Args:
			D (Optional[int]): Dimension of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, -(D ** 2), D ** 2, **kwargs)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: trid_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
