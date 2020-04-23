# encoding=utf8

"""Implementation of Easom benchmark."""

from math import pi

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import easom_function

__all__ = ["Easom"]


class Easom(Benchmark):
	r"""Implementations of Easom functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Easom Function

		:math:`f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			when :math:`a = 20`, :math:`b = 0.2` and :math:`c = 2 \pi` -> :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)}$

		Equation:
			\begin{equation} f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)} \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.Easom

	Attributes:
		Name (List[str]): Names for the benchmark.
		a (float): Function argument.
		b (float): Function argument.
		c (float): Function argument.
	"""
	Name = ["Easom"]

	def __init__(self, Lower=-100.0, Upper=100.0, a=20, b=0.2, c=2 * pi, **kwargs):
		r"""Initialize Easom benchmark.

		Args:
			Lower (Union[int, float, numpy.ndarray]): Lower bound of problem.
			Upper (Union[int, float, numpy.ndarray]): Upper bound of problem.
			a (float): Parameter of function.
			b (float): Parameter of function.
			c (float): Parameter of function.
			kwargs (Dict[str, Any]): Additional arguments for benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		self.a, self.b, self.c = a, b, c
		Benchmark.__init__(self, Lower, Upper, **kwargs)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)}$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, dict], float]: Evaluation function.
		"""
		return lambda x: easom_function(x, self.a, self.b, self.c)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
