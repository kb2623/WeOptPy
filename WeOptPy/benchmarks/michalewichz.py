# encoding=utf8

"""Implementation of Michalewichz"s benchmark."""

import numpy as np

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import michalewichz_function

__all__ = ["Michalewichz"]


class Michalewichz(Benchmark):
	r"""Implementations of Michalewichz's functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Michalewichz's Function

		:math:`f(\textbf{x}) = \sum_{i=1}^D \left( 10^6 \right)^{ \frac{i - 1}{D - 1} } x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, \pi]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			* at :math:`d = 2` :math:`f(\textbf{x}^*) = -1.8013` at :math:`\textbf{x}^* = (2.20, 1.57)`
			* at :math:`d = 5` :math:`f(\textbf{x}^*) = -4.687658`
			* at :math:`d = 10` :math:`f(\textbf{x}^*) = -9.66015`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$

		Equation:
			\begin{equation} f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m} \end{equation}

		Domain:
			:math:`0 \leq x_i \leq \pi`

	Reference URL:
		https://www.sfu.ca/~ssurjano/michal.html

	Attributes:
		Name (List[str]): Names for the benchmark.
		m (float): Function parameter.
	"""
	Name = ["Michalewichz"]
	m = 10.0

	def __init__(self, Lower=0.0, Upper=np.pi, m=10.0, **kwargs):
		r"""Initialize Michalewichz benchmark.

		Args:
			Lower (Union[int, float, numpy.ndarray]): Lower bound of problem.
			Upper (Union[int, float, numpy.ndarray]): Upper bound of problem.
			m (float): m attribute.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper, **kwargs)
		self.m = m

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: michalewichz_function(x, self.m)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
