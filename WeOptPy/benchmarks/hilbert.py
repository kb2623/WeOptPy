# encoding=utf8

"""Implementation of Hilbert benchmark."""

import numpy as np

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import Hilbert_function

__all__ = ["Hilbert"]


class Hilbert(Benchmark):
	r"""Implementations of Hilbert function.

	Find the inverse of the (ill-conditioned) Hilbert matrix
	* valid for any dimension, n=k*k, k=2,3,4,...
	* constraints: unconstrained
	* type: multi-modal with one global minimum; non-separable
	* initial upper bound = 2^n, initial lower bound = -2^n
	* value-to-reach = f(x*)+1.0e-8
	* f(x*) = 0.0; x*={{9,-36,30},{-36,192,-180},{30,-180,180}} (n=9)
	* x*={{16,-120,240,-140},{-120,1200,-2700,1680},{240,-2700,6480,4200},{-140,1680,-4200,2800}} (n=16)

	Date:
		April 2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Hilbert function

		:math:``

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-\infty, \infty]`, for all :math:`i = 1, 2,..., D`.

		LaTeX formats:
			Inline:
				$$

			Equation:
				\begin{equation} \end{equation}

			Domain:
				:math:``

	See Also:
		* :class:`NiaPy.benchmarks.Benchmark`
	"""
	Name = ["Hilbert"]

	def __init__(self, Lower=-np.inf, Upper=np.inf, **kwargs):
		r"""Create Hilbert benchmark.

		Args:
			Lower (Union[int, float, numpy.ndarray]): Lower bound limits.
			Upper (Union[int, float, numpy.ndarray]): Upper bound limits.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper, **kwargs)

	@staticmethod
	def latex_code():
		r"""Get latex code for function.

		Returns:
			str: Latex code.
		"""
		return r"""TODO"""

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda sol, **a: Hilbert_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
