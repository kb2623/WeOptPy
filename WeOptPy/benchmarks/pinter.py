# encoding=utf8

"""Implementation of Pinter benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import pinter_function

__all__ = ["Pinter"]


class Pinter(Benchmark):
	r"""Implementation of Pintér function.

	Date:
		2018

	Author:
		Klemen Brekovič

	License:
		MIT

	Function:
		Pintér function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i \log_{10} (1 + iB^2);` :math:`A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad`
		:math:`B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i \log_{10} (1 + iB^2); A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i \log_{10} (1 + iB^2); A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1) \end{equation}

		Domain:
			$-10 \leq x_i \leq 10$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Pinter"]

	def __init__(self, Lower=-10.0, Upper=10.0, **kwargs):
		r"""Initialize Pinter benchmark.

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
		return r''' $f(\mathbf{x}) = \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i \log_{10} (1 + iB^2); A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)$'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: pinter_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
