# encoding=utf8

"""Implementation of Lennard-Jones Minimum Energy Cluster benchmark."""

import numpy as np

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import Lennard_Jones_function

__all__ = ["LennardJones"]


class LennardJones(Benchmark):
	r"""Implementations of Lennard-Jones Minimum Energy Cluster function.

	Find the atomic configuration with minimum energy.
	* valid for any dimension, D=3*k, k=2,3,4,...,25.   k is the number of atoms in 3-D space
	* constraints: unconstrained
	* type: multi-modal with one global minimum; non-separable
	* initial upper bound = 4, initial lower bound = -4
	* value-to-reach = minima[k-2]+.0001
	* f(x*) = minima[k-2]; see array of minima below; additional minima available at the
	* Cambridge cluster database: http://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html

	Date:
		April 2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Lennard-Jones Minimum Energy Cluster

		:math:`f(\mathbf{x}) = \sum_{i=1}^{N-1}\sum_{j=i+1}^{N} \left( \frac{1}{d_{i,j}^2} - \frac{2}{d_{i,j}} \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-\infty, \infty]`, for all :math:`i = 1, 2,..., D`.

		LaTeX formats:
			Inline:
				$f(\mathbf{x}) = \sum_{i=1}^{N-1}\sum_{j=i+1}^{N} \left( \frac{1}{d_{i,j}^2} - \frac{2}{d_{i,j}} \right)$

			Equation:
				\begin{equation} f(\mathbf{x}) = \sum_{i=1}^{N-1}\sum_{j=i+1}^{N} \left( \frac{1}{d_{i,j}^2} - \frac{2}{d_{i,j}} \right) \end{equation}

			Domain:
				:math:`-\infty \leq x_i \leq \infty`

	See Also:
		* :class:`NiaPy.benchmarks.Benchmark`

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["LennardJones"]

	def __init__(self, Lower=-np.inf, Upper=np.inf, **kwargs):
		r"""Create Lennard Jones benchmakr.

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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^{N-1}\sum_{j=i+1}^{N} \left( \frac{1}{d_{i,j}^2} - \frac{2}{d_{i,j}} \right)$"""

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda sol, **a: Lennard_Jones_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
