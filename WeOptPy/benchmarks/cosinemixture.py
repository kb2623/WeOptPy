# encoding=utf8

"""Implementation of Cosine mixture benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import cosinemixture_function

__all__ = ["CosineMixture"]


class CosineMixture(Benchmark):
	r"""Implementations of Cosine mixture function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Cosine Mixture Function

		:math:`f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		Global maximu:
			:math:`f(x^*) = -0.1 D`, at :math:`x^* = (0.0,...,0.0)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$

		Equation:
			\begin{equation} f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2 \end{equation}

		Domain:
			$-1 \leq x_i \leq 14

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["CosineMixture"]

	def __init__(self, Lower=-1.0, Upper=1.0, **kwargs):
		r"""Initialize Cosine mixture benchmark.

		Args:
			Lower (Optional[Union[int, float, numpy.ndarray]]): Lower bound of problem.
			Upper (Optional[Union[int, float, numpy.ndarray]]): Upper bound of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper, **kwargs)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: cosinemixture_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
