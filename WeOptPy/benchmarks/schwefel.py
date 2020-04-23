# encoding=utf8

"""Implementations of Schwefels benchmarks."""

from WeOptPy.benchmarks.interfaces import Benchmark
from .functions import (
	modified_schwefel_function,
	schwefel_function,
	schwefel1222_function,
	schwefel1221_function
)

__all__ = [
	"Schwefel",
	"Schwefel221",
	"Schwefel222",
	"ModifiedSchwefel"
]


class Schwefel(Benchmark):
	r"""Implementation of Schewel function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Schwefel function

		:math:`f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

		LaTeX formats:
			Inline:
				$f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})$

			Equation:
				\begin{equation} f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|}) \end{equation}

		Domain:
			$-500 \leq x_i \leq 500$

	Reference:
		https://www.sfu.ca/~ssurjano/schwef.html

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Schwefel"]

	def __init__(self, Lower=-500.0, Upper=500.0, **kwargs):
		"""Initialize Schewefel benchmark.

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
		return r"""$f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: schwefel_function(x)


class Schwefel221(Benchmark):
	r"""Schwefel 2.21 function implementation.

	Date:
		2018

	Author:
		Klemen Berkovič

	Licence:
		MIT

	Function:
		Schwefel 2.21 function

		:math:`f(\mathbf{x})=\max_{i=1,...,D}|x_i|`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

		LaTeX formats:
			Inline:
				$f(\mathbf{x})=\max_{i=1,...,D}|x_i|$

			Equation:
				\begin{equation}f(\mathbf{x}) = \max_{i=1,...,D}|x_i| \end{equation}

		Domain:
			:math:`-100 \leq x_i \leq 100`

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name = ["Schwefel221"]

	def __init__(self, Lower=-100.0, Upper=100.0, **kwargs):
		"""Initialize Schwefel221 benchmark.

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
		return r'''$f(\mathbf{x})=\max_{i=1,...,D}|x_i|$'''

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x: schwefel1221_function(x)


class Schwefel222(Benchmark):
	r"""Schwefel 2.22 function implementation.

	Date:
		2018

	Author:
		Klemen Berkovič

	Licence:
		MIT

	Function:
		Schwefel 2.22 function

		:math:`f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|$

		Equation:
			\begin{equation}f(\mathbf{x}) = \sum_{i=1}^{D}|x_i| + \prod_{i=1}^{D}|x_i| \end{equation}

		Domain:
			:math:`-100 \leq x_i \leq 100`

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["Schwefel222"]

	def __init__(self, Lower=-100.0, Upper=100.0, **kwargs):
		"""Initialize Schwefel222 benchmark.

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
		return r"""$f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x, **a: schwefel1222_function(x)


class ModifiedSchwefel(Benchmark):
	r"""Implementations of Modified Schwefel functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Modified Schwefel Function

		:math:`f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}$

		Equation:
			\begin{equation} f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases} \end{equation}


		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf

	Attributes:
		Name (List[str]): Names for the benchmark.
	"""
	Name = ["ModifiedSchwefel"]

	def __init__(self, Lower=-100.0, Upper=100.0, **kwargs):
		"""Initialize Modified Schwefel benchmark.

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
		return r"""$f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}$"""

	def function(self):
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
		"""
		return lambda x: modified_schwefel_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
