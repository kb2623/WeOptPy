# encoding=utf8

"""Implementation of Stepint benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import stepint_function

__all__ = ["Stepint"]


class Stepint(Benchmark):
	r"""Implementation of Stepint functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Stepint function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (-5.12,...,-5.12)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$

		Equation:
			\begin{equation}f(\mathbf{x}) = \sum_{i=1}^D x_i^2 \end{equation}

		Domain:
			$0 \leq x_i \leq 10$

	Reference paper: Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Stepint"]

	def __init__(self, Lower=-5.12, Upper=5.12, **kwargs):
		"""Initialize Stepint benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: stepint_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
