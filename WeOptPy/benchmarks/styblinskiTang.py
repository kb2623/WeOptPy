# encoding=utf8

"""Implementation of Styblinski Tang benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import styblinskitang_function

__all__ = ["StyblinskiTang"]


class StyblinskiTang(Benchmark):
	r"""Implementation of Styblinski-Tang functions.

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Function:
		Styblinski-Tang function

		:math:`f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5, 5]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = -78.332`, at :math:`x^* = (-2.903534,...,-2.903534)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) $

		Equation:
			\begin{equation}f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) \end{equation}

		Domain:
			$-5 \leq x_i \leq 5$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["StyblinskiTang"]

	def __init__(self, Lower=-5.0, Upper=5.0, **kwargs):
		r"""Initialize Styblinski Tang benchmark.

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
		return r"""$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) $"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: styblinskitang_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
