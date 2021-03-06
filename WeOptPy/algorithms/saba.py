# encoding=utf8

"""Adaptive bat algorithm module."""

import numpy as np

from WeOptPy.algorithms.interfaces import Algorithm

__all__ = [
	'AdaptiveBatAlgorithm',
	'SelfAdaptiveBatAlgorithm'
]


class AdaptiveBatAlgorithm(Algorithm):
	r"""Implementation of Adaptive bat algorithm.

	Algorithm:
		Adaptive bat algorithm

	Date:
		April 2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		epsilon (float): Scaling factor.
		alpha (float): Constant for updating loudness.
		r (float): Pulse rate.
		Qmin (float): Minimum frequency.
		Qmax (float): Maximum frequency.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['AdaptiveBatAlgorithm', 'ABA']

	@staticmethod
	def type_parameters():
		r"""Return dict with where key of dict represents parameter name and values represent checking functions for selected parameter.

		Returns:
			Dict[str, Callable]:
				* epsilon (Callable[[Union[float, int]], bool]): Scale factor.
				* alpha (Callable[[Union[float, int]], bool]): Constant for updating loudness.
				* r (Callable[[Union[float, int]], bool]): Pulse rate.
				* Qmin (Callable[[Union[float, int]], bool]): Minimum frequency.
				* Qmax (Callable[[Union[float, int]], bool]): Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'epsilon': lambda x: isinstance(x, (float, int)) and x > 0,
			'alpha': lambda x: isinstance(x, (float, int)) and x > 0,
			'r': lambda x: isinstance(x, (float, int)) and x > 0,
			'Qmin': lambda x: isinstance(x, (float, int)),
			'Qmax': lambda x: isinstance(x, (float, int))
		})
		return d

	def set_parameters(self, n=100, a=0.5, epsilon=0.001, alpha=1.0, r=0.5, qmin=0.0, qmax=2.0, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			n (int): Number of individuals in population.
			a (Optional[float]): Starting loudness.
			epsilon (Optional[float]): Scaling factor.
			alpha (Optional[float]): Constant for updating loudness.
			r (Optional[float]): Pulse rate.
			qmin (Optional[float]): Minimum frequency.
			qmax (Optional[float]): Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)
		self.A, self.epsilon, self.alpha, self.r, self.Qmin, self.Qmax = a, epsilon, alpha, r, qmin, qmax

	def get_parameters(self):
		r"""Get algorithm parameters.

		Returns:
			Dict[str, Any]: Arguments values.

		See Also:
			* :func:`WeOptPy.algorithms.interfaces.Algorithm.getParameters`
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'a': self.A,
			'epsilon': self.epsilon,
			'alpha': self.alpha,
			'r': self.r,
			'Qmin': self.Qmin,
			'Qmax': self.Qmax
		})
		return d

	def init_population(self, task):
		r"""Initialize the starting population.

		Parameters:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* a (float): Loudness.
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray): 	TODO
					* v (numpy.ndarray): TODO

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.initPopulation`
		"""
		sol, fitness, args, d = Algorithm.init_population(self, task)
		a, s, Q, v = np.full(self.NP, self.A), np.full([self.NP, task.D], 0.0), np.full(self.NP, 0.0), np.full([self.NP, task.D], 0.0)
		d.update({'a': a, 'S': s, 'Q': Q, 'v': v})
		return sol, fitness, args, d

	def local_search(self, best, a, task, **kwargs):
		r"""Improve the best solution according to the Yang (2010).

		Args:
			best (numpy.ndarray): Global best individual.
			a (float): Loudness.
			task (Task): Optimization task.
			kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray: New solution based on global best individual.
		"""
		return task.repair(best + self.epsilon * a * self.normal(0, 1, task.D), rnd=self.Rand)

	def update_loudness(self, a):
		r"""Update loudness when the prey is found.

		Args:
			a (float): Loudness.

		Returns:
			float: New loudness.
		"""
		na = a * self.alpha
		return na if na > 1e-13 else self.A

	def run_iteration(self, task, Sol, Fitness, xb, fxb, a, S, Q, v, *args, **dparams):
		r"""Core function of Bat Algorithm.

		Parameters:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population.
			Fitness (numpy.ndarray[float]): Current population fitness/function values.
			xb (numpy.ndarray): Current best individual.
			fxb (float): Current best individual function/fitness value.
			a (numpy.ndarray): TODO
			S (numpy.ndarray): TODO
			Q (numpy.ndarray[float]): TODO
			v (numpy.ndarray[float]): TODO
			args (list): Additional arguments.
			dparams (dict): Additional keyword algorithm arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* a (numpy.ndarray): Loudness.
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray): TODO
					* v (numpy.ndarray): TODO
		"""
		for i in range(self.NP):
			Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1)
			v[i] += (Sol[i] - xb) * Q[i]
			if self.rand() > self.r: S[i] = self.local_search(best=xb, a=a[i], task=task, i=i, Sol=Sol)
			else: S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
			Fnew = task.eval(S[i])
			if (Fnew <= Fitness[i]) and (self.rand() < a[i]): Sol[i], Fitness[i] = S[i], Fnew
			if Fnew <= fxb: xb, fxb, a[i] = S[i].copy(), Fnew, self.update_loudness(a[i])
		return Sol, Fitness, xb, fxb, args, {'a': a, 'S': S, 'Q': Q, 'v': v}


class SelfAdaptiveBatAlgorithm(AdaptiveBatAlgorithm):
	r"""Implementation of Hybrid bat algorithm.

	Algorithm:
		Hybrid bat algorithm

	Date:
		April 2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "a Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		A_l (Optional[float]): lower limit of loudness.
		A_u (Optional[float]): upper limit of loudness.
		r_l (Optional[float]): lower limit of pulse rate.
		r_u (Optional[float]): upper limit of pulse rate.
		tao_1 (Optional[float]): Learning rate for loudness.
		tao_2 (Optional[float]): Learning rate for pulse rate.

	See Also:
		* :class:`NiaPy.algorithms.basic.BatAlgorithm`
	"""
	Name = ['SelfAdaptiveBatAlgorithm', 'SABA']

	@staticmethod
	def algorithm_info():
		r"""Get basic information about the algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "a Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7."""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: TODO

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.typeParameters`
		"""
		d = AdaptiveBatAlgorithm.type_parameters()
		d.pop('a', None), d.pop('r', None)
		d.update({
			'A_l': lambda x: isinstance(x, (float, int)) and x >= 0,
			'A_u': lambda x: isinstance(x, (float, int)) and x >= 0,
			'r_l': lambda x: isinstance(x, (float, int)) and x >= 0,
			'r_u': lambda x: isinstance(x, (float, int)) and x >= 0,
			'tao_1': lambda x: isinstance(x, (float, int)) and 0 <= x <= 1,
			'tao_2': lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		})
		return d

	def set_parameters(self, A_l=0.9, A_u=1.0, r_l=0.001, r_u=0.1, tao_1=0.1, tao_2=0.1, **ukwargs):
		r"""Set core parameters of HybridBatAlgorithm algorithm.

		Arguments:
			A_l (Optional[float]): lower limit of loudness.
			A_u (Optional[float]): upper limit of loudness.
			r_l (Optional[float]): lower limit of pulse rate.
			r_u (Optional[float]): upper limit of pulse rate.
			tao_1 (Optional[float]): Learning rate for loudness.
			tao_2 (Optional[float]): Learning rate for pulse rate.

		See Also:
			* :func:`NiaPy.algorithms.modified.AdaptiveBatAlgorithm.setParameters`
		"""
		AdaptiveBatAlgorithm.set_parameters(self, **ukwargs)
		self.A_l, self.A_u, self.r_l, self.r_u, self.tao_1, self.tao_2 = A_l, A_u, r_l, r_u, tao_1, tao_2

	def get_parameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]: Parameters of the algorithm.

		See Also:
			* :func:`NiaPy.algorithms.modified.AdaptiveBatAlgorithm.getParameters`
		"""
		d = AdaptiveBatAlgorithm.get_parameters(self)
		d.update({
			'A_l': self.A_l,
			'A_u': self.A_u,
			'r_l': self.r_l,
			'r_u': self.r_u,
			'tao_1': self.tao_1,
			'tao_2': self.tao_2
		})
		return d

	def init_population(self, task):
		r"""Initialize initial population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* a (numpy.ndarray): Populations loudness.
					* r (numpy.ndarray): TODO
		"""
		Sol, Fitness, args, d = AdaptiveBatAlgorithm.init_population(self, task)
		A, r = np.full(self.NP, self.A), np.full(self.NP, self.r)
		d.update({'a': A, 'r': r})
		return Sol, Fitness, args, d

	def self_adaptation(self, a, r):
		r"""Adaptation step.

		Args:
			a (float): Current loudness.
			r (float): Current pulse rate.

		Returns:
			Tuple[float, float]:
				1. New loudness.
				2. Nwq pulse rate.
		"""
		return self.A_l + self.rand() * (self.A_u - self.A_l) if self.rand() < self.tao_1 else a, self.r_l + self.rand() * (self.r_u - self.r_l) if self.rand() < self.tao_2 else r

	def run_iteration(self, task, Sol, Fitness, xb, fxb, a, r, S, Q, v, *args, **dparams):
		r"""Core function of Bat Algorithm.

		Parameters:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population
			Fitness (numpy.ndarray[float]): Current population fitness/function values
			xb (numpy.ndarray): Current best individual
			fxb (float): Current best individual function/fitness value
			a (numpy.ndarray): Loudness of individuals.
			r (numpy.ndarray): Pulse rate of individuals.
			S (numpy.ndarray): TODO
			Q (numpy.ndarray): TODO
			v (numpy.ndarray): TODO
			args (list): Additional arguments.
			dparams (dict): Additional algorithm keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments:
					* a (numpy.ndarray): Loudness.
					* r (numpy.ndarray): Pulse rate.
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray): TODO
					* v (numpy.ndarray): TODO
		"""
		for i in range(self.NP):
			a[i], r[i] = self.self_adaptation(a[i], r[i])
			Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1)
			v[i] += (Sol[i] - xb) * Q[i]
			if self.rand() > r[i]: S[i] = self.local_search(best=xb, a=a[i], task=task, i=i, Sol=Sol, Fitness=Fitness)
			else: S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
			Fnew = task.eval(S[i])
			if (Fnew <= Fitness[i]) and (self.rand() < (self.A_l - a[i]) / self.A): Sol[i], Fitness[i] = S[i], Fnew
			if Fnew <= fxb: xb, fxb = S[i].copy(), Fnew
		return Sol, Fitness, xb, fxb, args, {'a': a, 'r': r, 'S': S, 'Q': Q, 'v': v}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
