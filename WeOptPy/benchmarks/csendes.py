# encoding=utf8

"""Implementation of Csendes benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import csendes_function

__all__ = ["Csendes"]


class Csendes(Benchmark):
	r"""Implementation of Csendes function.

	Date:
		2018

	Author:
		Klemen Brekovič

	License:
		MIT

	Function:
		Csendes function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^6\left( 2 + \sin \frac{1}{x_i}\right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D x_i^6\left( 2 + \sin \frac{1}{x_i}\right)$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^D x_i^6\left( 2 + \sin \frac{1}{x_i}\right) \end{equation}

		Domain:
			:math:`-1 \leq x_i \leq 1`

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Csendes"]

	def __init__(self, Lower=-1.0, Upper=1.0, **kwargs):
		r"""Initialize Csendes benchmark.

		Args:
			Lower (Union[int, float, np.ndarray]): Lower bound of problem.
			Upper (Union[int, float, np.ndarray]): Upper bound of problem.
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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^D x_i^6\left( 2 + \sin \frac{1}{x_i}\right)$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[np.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: csendes_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
