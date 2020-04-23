# encoding=utf8

"""Implementation of DeflectedCorrugatedSpring benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import deflected_corrugated_spring_function

__all__ = ["DeflectedCorrugatedSpring"]


class DeflectedCorrugatedSpring(Benchmark):
	r"""Implementations of Deflected Corrugated Spring functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Deflected Corrugated Spring Function

		:math:`f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, 2 \pi]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			when :math:`K = 5` and :math:`\alpha = 5` -> :math:`f(x^*) = -1`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right)$

		Equation:
			\begin{equation} f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right) \end{equation}

		Domain:
			$0 \leq x_i \leq 2 \pi$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.DeflectedCorrugatedSpring

	Attributes:
		Name (List[str]): Names for the benchmark.
		alpha (float): Function parameter.
		K (float): function parameter.
	"""
	Name = ["DeflectedCorrugatedSpring"]

	def __init__(self, Lower=-0.0, alpha=5.0, K=5.0, **kwargs):
		r"""Initialize HGBat benchmark.

		Args:
			Lower (Optional[Union[int, float, numpy.ndarray]]): Lower bound of problem.
			alpha (Optional[Union[int, float, numpy.ndarray]]): Parameter for function and upper limit.
			K (Optional[float]): Parameter of function.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, 2 * alpha, **kwargs)
		self.alpha, self.K = alpha, K

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right)$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: deflected_corrugated_spring_function(x, self.alpha, self.K)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
