# encoding=utf8

"""Harmony search algorithm module."""

import numpy as np

from WeOptPy.algorithms.interfaces.algorithm import Algorithm

__all__ = [
	"HarmonySearch",
	"HarmonySearchV1"
]


class HarmonySearch(Algorithm):
	r"""Implementation of harmony search algorithm.

	Algorithm:
		Harmony Search Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://link.springer.com/chapter/10.1007/978-3-642-00185-7_1

	Reference paper:
		Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14.

	Attributes:
		Name (List[str]): List of strings representing algorithm names
		r_accept (float): Probability of accepting new bandwidth into harmony.
		r_pa (float): Probability of accepting random bandwidth into harmony.
		b_range (float): Range of bandwidth.

	See Also:
		* :class:`WeOptPy.algorithms.Algorithm`
	"""
	Name = ["HarmonySearch", "HS"]

	@staticmethod
	def algorithm_info():
		r"""Get basic information about the algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* HMS (Callable[[int], bool])
				* r_accept (Callable[[float], bool])
				* r_pa (Callable[[float], bool])
				* b_range (Callable[[float], bool])
		"""
		return {
			"HMS": lambda x: isinstance(x, int) and x > 0,
			"r_accept": lambda x: isinstance(x, float) and 0 < x < 1,
			"r_pa": lambda x: isinstance(x, float) and 0 < x < 1,
			"b_range": lambda x: isinstance(x, (int, float)) and x > 0
		}

	def set_parameters(self, HMS=30, r_accept=0.7, r_pa=0.35, b_range=1.42, **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
			HMS (Optional[int]): Number of harmony in the memory
			r_accept (Optional[float]): Probability of accepting new bandwidth to harmony.
			r_pa (Optional[float]): Probability of accepting random bandwidth into harmony.
			b_range (Optional[float]): Bandwidth range.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('n', None)
		Algorithm.set_parameters(self, n=HMS, **ukwargs)
		self.r_accept, self.r_pa, self.b_range = r_accept, r_pa, b_range

	def get_parameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
				* Parameter name: Represents a parameter name
				* Value of parameter: Represents the value of the parameter
		"""
		d = Algorithm.get_parameters(self)
		d.pop('n', None)
		d.update({
			'HMS': self.NP,
			'r_accept': self.r_accept,
			'r_pa': self.r_pa,
			'b_range': self.b_range
		})
		return d

	def bw(self, task):
		r"""Get bandwidth.

		Args:
			task (Task): Optimization task.

		Returns:
			float: Bandwidth.
		"""
		return self.uniform(-1, 1) * self.b_range

	def adjustment(self, x, task):
		r"""Adjust value based on bandwidth.

		Args:
			x (Union[int, float]): Current position.
			task (Task): Optimization task.

		Returns:
			float: New position.
		"""
		return x + self.bw(task)

	def improvize(self, HM, task):
		r"""Create new individual.

		Args:
			HM (numpy.ndarray): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: New individual.
		"""
		H = np.full(task.D, .0)
		for i in range(task.D):
			r, j = self.rand(), self.randint(self.NP)
			H[i] = HM[j, i] if r > self.r_accept else self.adjustment(HM[j, i], task) if r > self.r_pa else self.uniform(task.Lower[i], task.Upper[i])
		return H

	def init_population(self, task):
		r"""Initialize first population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New harmony/population.
				2. New population fitness/function values.
				3. Additional parameters.
				4. Additional keyword parameters.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.initPopulation`
		"""
		return Algorithm.init_population(self, task)

	def run_iteration(self, task, HM, HM_f, xb, fxb, *args, **dparams):
		r"""Core function of HarmonySearch algorithm.

		Args:
			task (Task): Optimization task.
			HM (numpy.ndarray): Current population.
			HM_f (numpy.ndarray): Current populations function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best fitness/function value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New harmony/population.
				2. New populations function/fitness values.
				3. New global best solution
				4. New global best solution fitness/objective value
				5. Additional arguments.
				6. Additional keyword arguments.
		"""
		H = self.improvize(HM, task)
		H_f = task.eval(task.repair(H, self.Rand))
		iw = np.argmax(HM_f)
		if H_f <= HM_f[iw]: HM[iw], HM_f[iw] = H, H_f
		xb, fxb = self.get_best(H, H_f, xb, fxb)
		return HM, HM_f, xb, fxb, args, dparams


class HarmonySearchV1(HarmonySearch):
	r"""Implementation of harmony search algorithm.

	Algorithm:
		Harmony Search Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://link.springer.com/chapter/10.1007/978-3-642-00185-7_1

	Reference paper:
		Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		bw_min (float): Minimal bandwidth.
		bw_max (float): Maximal bandwidth.

	See Also:
		* :class:`WeOptPy.algorithms.HarmonySearch`
	"""
	Name = ["HarmonySearchV1", "HSv1"]

	@staticmethod
	def algorithm_info():
		r"""Get basic information about algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Function for testing correctness of parameters.

		See Also:
			* :func:`WeOptPy.algorithms.HarmonySearch.typeParameters`
		"""
		d = HarmonySearch.type_parameters()
		del d["b_range"]
		d.update({
			"dw_min": lambda x: isinstance(x, (float, int)) and x >= 1,
			"dw_max": lambda x: isinstance(x, (float, int)) and x >= 1
		})
		return d

	def set_parameters(self, bw_min=1, bw_max=2, **kwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			bw_min (Optional[float]): Minimal bandwidth
			bw_max (Optional[float]): Maximal bandwidth
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.hs.HarmonySearch.setParameters`
		"""
		HarmonySearch.set_parameters(self, **kwargs)
		self.bw_min, self.bw_max = bw_min, bw_max

	def get_parameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
				* Parameter name: Represents a parameter name
				* Value of parameter: Represents the value of the parameter
		"""
		d = HarmonySearch.get_parameters(self)
		d.update({
			'bw_min': self.bw_min,
			'bw_max': self.bw_max
		})
		return d

	def bw(self, task):
		r"""Get new bandwidth.

		Args:
			task (Task): Optimization task.

		Returns:
			float: New bandwidth.
		"""
		return self.bw_min * np.exp(np.log(self.bw_min / self.bw_max) * task.Iters / task.nGEN)
