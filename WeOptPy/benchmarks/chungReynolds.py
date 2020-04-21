# encoding=utf8

"""Implementation of Chung Reynolds benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import chungreynolds_function

__all__ = ["ChungReynolds"]


class ChungReynolds(Benchmark):
	r"""Implementation of Chung Reynolds functions.

	Date:
		2018

	Authors:
		Klemen Brekovič

	License:
		MIT

	Function:
		Chung Reynolds function

		:math:`f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2$

		Equation:
			\begin{equation} f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2 \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names of the benchmark.
	"""
	Name = ["ChungReynolds"]

	def __init__(self, Lower=-100.0, Upper=100.0, **kwargs):
		"""Initialize Chung Reynolds benchmark.

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
		return r'''$f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2$'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[np.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: chungreynolds_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
