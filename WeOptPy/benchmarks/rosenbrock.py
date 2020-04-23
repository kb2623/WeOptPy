# encoding=utf8

"""Implementation of Rosenbrock benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import rosenbrock_function

__all__ = ["Rosenbrock"]

class Rosenbrock(Benchmark):
	r"""Implementation of Rosenbrock benchmark function.

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Function:
		Rosenbrock function

		:math:`f(\mathbf{x}) = \sum_{i=1}^{D-1} \left (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-30, 30]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (1,...,1)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2) \end{equation}

		Domain:
			$-30 \leq x_i \leq 30$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Rosenbrock"]

	def __init__(self, Lower=-30.0, Upper=30.0, **kwargs):
		"""Initialize Rosenbrock benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)$'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda sol, **a: rosenbrock_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
