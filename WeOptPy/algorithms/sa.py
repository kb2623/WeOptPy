# encoding=utf8

import numpy as np

from WeOptPy.algorithms.interfaces import Algorithm

__all__ = [
	'SimulatedAnnealing',
	'coolDelta',
	'coolLinear'
]


def coolDelta(currentT, T, deltaT, nFES, **kwargs):
	r"""Calculate new temperature by differences.

	Args:
		currentT (float):
		T (float):
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		float: New temperature.
	"""
	return currentT - deltaT


def coolLinear(currentT, T, deltaT, nFES, **kwargs):
	r"""Calculate temperature with linear function.

	Args:
		currentT (float): Current temperature.
		T (float):
		deltaT (float):
		nFES (int): Number of evaluations done.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		float: New temperature.
	"""
	return currentT - T / nFES


class SimulatedAnnealing(Algorithm):
	r"""Implementation of Simulated Annealing Algorithm.

	Algorithm:
		Simulated Annealing Algorithm

	Date:
		2018

	Authors:
		Jan Popič and Klemen Berkovič

	License:
		 MIT

	Reference URL:
		TODO

	Reference paper:
		TODO

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		delta (float): Movement for neighbour search.
		T (float); Starting temperature.
		deltaT (float): Change in temperature.
		coolingMethod (Callable): Neighbourhood function.
		epsilon (float): Error value.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['SimulatedAnnealing', 'SA']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""None"""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* delta (Callable[[Union[float, int], bool]): TODO
		"""
		return {
			'delta': lambda x: isinstance(x, (int, float)) and x > 0,
			'T': lambda x: isinstance(x, (int, float)) and x > 0,
			'deltaT': lambda x: isinstance(x, (int, float)) and x > 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		}

	def set_parameters(self, delta=0.5, T=2000, deltaT=0.8, coolingMethod=coolDelta, epsilon=1e-23, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		Arguments:
			delta (Optional[float]): Movement for neighbour search.
			T (Optional[float]); Starting temperature.
			deltaT (Optional[float]): Change in temperature.
			coolingMethod (Optional[Callable]): Neighbourhood function.
			epsilon (Optional[float]): Error value.

		See Also
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('n', None)
		Algorithm.set_parameters(self, n=1, **ukwargs)
		self.delta, self.T, self.deltaT, self.cool, self.epsilon = delta, T, deltaT, coolingMethod, epsilon

	def get_parameters(self):
		r"""Get algorithms parametes values.

		Returns:
			Dict[str, Any]:

		See Also
			* :func:`NiaPy.algorithms.Algorithm.getParameters`
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'delta': self.delta,
			'deltaT': self.deltaT,
			'T': self.T,
			'epsilon': self.epsilon
		})
		return d

	def init_population(self, task):
		r"""Initialize the starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, dict]:
				1. Initial solution
				2. Initial solutions fitness/objective value
				3. Additional arguments
		"""
		x = task.lower + task.range() * self.rand(task.D)
		curT, xfit = self.T, task.eval(x)
		return x, xfit, {'curT': curT}

	def run_iteration(self, task, x, xfit, xb, fxb, curT, **dparams):
		r"""Core funciton of the algorithm.

		Args:
			task (Task):
			x (numpy.ndarray):
			xfit (float):
			xb (numpy.ndarray):
			fxb (float):
			curT (float):
			dparams (dict): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray, float, dict]:
				1. New solution
				2. New solutions fitness/objective value
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Additional arguments
		"""
		c = task.repair(x - self.delta / 2 + self.rand(task.D) * self.delta, rnd=self.Rand)
		cfit = task.eval(c)
		deltaFit, r = cfit - xfit, self.rand()
		if deltaFit < 0 or r < np.exp(deltaFit / curT): x, xfit = c, cfit
		curT = self.cool(curT, self.T, deltaT=self.deltaT, nFES=task.nFES)
		xb, fxb = self.get_best(x, xfit, xb, fxb)
		return x, xfit, xb, fxb, {'curT': curT}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
