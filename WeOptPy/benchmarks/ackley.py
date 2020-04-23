# encoding=utf8

"""The module implementing Ackley benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import ackley_function

__all__ = ["Ackley"]


class Ackley(Benchmark):
	r"""Implementation of Ackley function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Ackley function

		:math:`f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D}\sum_{i=1}^D \cos(c\;x_i)\right) + a + \exp(1)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-32.768, 32.768]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(\textbf{x}^*) = 0`, at  :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D} \sum_{i=1}^D cos(c\;x_i)\right) + a + \exp(1)$

		Equation:
			\begin{equation}f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D} \sum_{i=1}^D \cos(c\;x_i)\right) + a + \exp(1) \end{equation}

		Domain:
			$-32.768 \leq x_i \leq 32.768$

	Reference:
		https://www.sfu.ca/~ssurjano/ackley.html

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Ackley"]

	def __init__(self, Lower=-32.768, Upper=32.768, **kwargs):
		"""Initialize Ackley benchmark.

		Args:
			Lower (Optional[Union[int, float, np.ndarray]]): Lower bound of problem.
			Upper (Optional[Union[int, float, np.ndarray]]): Upper bound of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper, **kwargs)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: latex code.
		"""
		return r"""$f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D} \sum_{i=1}^D cos(c\;x_i)\right) + a + \exp(1)$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[np.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: ackley_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
