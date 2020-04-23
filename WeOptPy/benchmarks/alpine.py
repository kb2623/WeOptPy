# encoding=utf8

"""Implementations of Alpine benchmarks."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import (
	alpine1_function,
	alpine2_function
)

__all__ = [
	"Alpine1",
	"Alpine2"
]


class Alpine1(Benchmark):
	r"""Implementation of Alpine1 function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Alpine1 function

		:math:`f(\mathbf{x}) = \sum_{i=1}^{D} |x_i \sin(x_i)+0.1x_i|`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^{D} \left |x_i \sin(x_i)+0.1x_i \right|$

		Equation:
			\begin{equation} f(x) = \sum_{i=1}^{D} \left|x_i \sin(x_i) + 0.1x_i \right| \end{equation}

		Domain:
			$`-10 \leq x_i \leq 10$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names of the benchmark
	"""
	Name = ["Alpine1"]

	def __init__(self, Lower=-10.0, Upper=10.0, **kwargs):
		r"""Initialize of Alpine1 benchmark.

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
			str: Latex code.
		"""
		return r"""$f(\mathbf{x}) = \sum_{i=1}^{D} \left |x_i \sin(x_i)+0.1x_i \right|$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[np.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: alpine1_function(x)


class Alpine2(Benchmark):
	r"""Implementation of Alpine2 function.

	Date:
		2018

	Author:
		Klemen Brekovič

	License:
		MIT

	Function:
		Alpine2 function

		:math:`f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 2.808^D`, at :math:`x^* = (7.917,...,7.917)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)$

		Equation:
			\begin{equation} f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i) \end{equation}

		Domain:
			$0 \leq x_i \leq 10$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names of the benchmark.
	"""
	Name = ["Alpine2"]

	def __init__(self, Lower=0.0, Upper=10.0, **kwargs):
		r"""Initialize Alpine2 benchmark.

		Args:
			Lower (Union[int, float, np.ndarray]): Lower bound of problem.
			Upper (Union[int, float, np.ndarray]): Upper bound of problem.
			kwargs (Dict[str, Any]): Additional arguments for the benchmark.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower=Lower, Upper=Upper, **kwargs)

	@staticmethod
	def latex_code():
		"""Return the latex code of the problem.

		Returns:
			str: Latex code.
		"""
		return r"""$f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[np.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: alpine2_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
