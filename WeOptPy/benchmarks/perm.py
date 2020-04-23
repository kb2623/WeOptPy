# encoding=utf8

"""Implementation of Perm benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import perm_function

__all__ = ["Perm"]


class Perm(Benchmark):
	r"""Implementations of Perm functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Arguments:
		beta (real): Value added to inner sum of function.

	Function:
		Perm Function

		:math:`f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-D, D]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \frac{1}{2}, \cdots , \frac{1}{i} , \cdots , \frac{1}{D})`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2 \end{equation}

		Domain:
			$-D \leq x_i \leq D$

	Reference:
		https://www.sfu.ca/~ssurjano/perm0db.html

	Attributes:
		Name (List[str]): Names for the benchmark.
		beta (float): Function parameter.
	"""
	Name = ["Perm"]

	def __init__(self, D=10, beta=0.5, **kwargs):
		"""Initialize Perm benchmark.

		Args:
			D (Optional[int]): Dimension on problem.
			beta (Optional[float]): Beta parameter.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, -D, D, **kwargs)
		self.beta = beta

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: perm_function(x, self.beta)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
