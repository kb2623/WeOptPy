# encoding=utf8

"""Implementation of Happy Cat benchmark."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import happycat_function

__all__ = ["HappyCat"]


class HappyCat(Benchmark):
	r"""Implementation of Happy cat function.

	Date:
		2018

	Author:
		Lucija Brezočnik

	License:
		MIT

	Function:
		Happy cat function

		:math:`f(\mathbf{x}) = {\left |\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (-1,...,-1)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5$

		Equation:
			\begin{equation} f(\mathbf{x}) = {\left| \sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5 \end{equation}

		Domain:
			:math:`-100 \leq x_i \leq 100`

	Reference URL:
		http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf

	Reference:
		Beyer, H. G., & Finck, S. (2012). HappyCat - A Simple Function Class Where Well-Known Direct Search Algorithms Do Fail. In International Conference on Parallel Problem Solving from Nature (pp. 367-376). Springer, Berlin, Heidelberg.

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["HappyCat"]

	def __init__(self, Lower=-100.0, Upper=100.0, **kwargs):
		"""Initialize Happy Cat benchmark.

		Args:
			Lower (Union[int, float, numpy.ndarray]): Lower bound of problem.
			Upper (Union[int, float, numpy.ndarray]): Upper bound of problem.
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
		return r'''$f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5$'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda sol, **a: happycat_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
