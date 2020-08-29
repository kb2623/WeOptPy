# encoding=utf8

import numpy as np
from scipy.stats import levy

from WeOptPy.algorithms.interfaces import Algorithm

__all__ = ['CuckooSearch']


class CuckooSearch(Algorithm):
	r"""Implementation of Cuckoo behaviour and levy flights.

	Algorithm:
		Cuckoo Search

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference:
		Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009.

	Attributes:
		Name (List[str]): list of strings representing algorithm names.
		N (int): Population size.
		pa (float): Proportion of worst nests.
		alpha (float): Scale factor for levy flight.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['CuckooSearch', 'CS']

	@staticmethod
	def algorithm_info():
		r"""Get algorithms information.

		Returns:
			str: Algorithm information.
		"""
		return r"""Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009."""

	@staticmethod
	def type_parameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* N (Callable[[int], bool]): TODO
				* pa (Callable[[float], bool]): TODO
				* alpha (Callable[[Union[int, float]], bool]): TODO
		"""
		return {
			'N': lambda x: isinstance(x, int) and x > 0,
			'pa': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'alpha': lambda x: isinstance(x, (float, int)),
		}

	def set_parameters(self, N=50, pa=0.2, alpha=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			N (int): Population size :math:`\in [1, \infty)`
			pa (float): factor :math:`\in [0, 1]`
			alpah (float): TODO
			ukwargs (Dict[str, Any]): Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('n', None)
		Algorithm.set_parameters(self, n=N, **ukwargs)
		self.pa, self.alpha = pa, alpha

	def get_parameters(self):
		d = Algorithm.get_parameters(self)
		d.pop('n', None)
		d.update({
			'N': self.NP,
			'pa': self.pa,
			'alpha': self.alpha
		})
		return d

	def empty_nests(self, pop, fpop, pa_v, task):
		r"""Empty nests.

		Args:
			pop (numpy.ndarray): Current population
			fpop (numpy.ndarray[float]): Current population fitness/funcion values
			pa_v (): TODO.
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. New population
				2. New population fitness/function values
		"""
		si = np.argsort(fpop)[:int(pa_v):-1]
		pop[si] = task.Lower + self.rand(task.D) * task.bRange
		fpop[si] = np.apply_along_axis(task.eval, 1, pop[si])
		return pop, fpop

	def init_population(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. Additional arguments:
				4. Additional keyword arguments:
					* pa_v (float): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		N, N_f, args, d = Algorithm.init_population(self, task)
		d.update({'pa_v': self.NP * self.pa})
		return N, N_f, args, d

	def run_iteration(self, task, pop, fpop, xb, fxb, pa_v, *args, **dparams):
		r"""Core function of CuckooSearch algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individual function/fitness values.
			pa_v (float): TODO
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Additional arguments:
				6. Additional keyword arguments:
					* pa_v (float): TODO
		"""
		i = self.randint(self.NP)
		Nn = task.repair(pop[i] + self.alpha * levy.rvs(size=[task.D], random_state=self.Rand), rnd=self.Rand)
		Nn_f = task.eval(Nn)
		j = self.randint(self.NP)
		while i == j: j = self.randint(self.NP)
		if Nn_f <= fpop[j]: pop[j], fpop[j] = Nn, Nn_f
		pop, fpop = self.empty_nests(pop, fpop, pa_v, task)
		xb, fxb = self.get_best(pop, fpop, xb, fxb)
		return pop, fpop, xb, fxb, args, {'pa_v': pa_v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
