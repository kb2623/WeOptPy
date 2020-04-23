# encoding=utf8

"""Implementation of Rastrigin benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import rastrigin_function

__all__ = ["Rastrigin"]


class Rastrigin(Benchmark):
	r"""Implementation of Rastrigin benchmark function.

	Date:
		2018

	Authors:
		Klemen Brekovič

	License:
		MIT

	Function:
		Rastrigin function

		:math:`f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$

		Equation:
			\begin{equation} f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right) \end{equation}

		Domain:
			$-5.12 \leq x_i \leq 5.12$

	Reference:
		https://www.sfu.ca/~ssurjano/rastr.html

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ['Rastrigin']

	def __init__(self, Lower=-5.12, Upper=5.12, **kwargs):
		"""Initialize Rastrigin benchmark.

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
		return r'''$f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda sol, **a: rastrigin_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
