# encoding=utf8

import numpy as np
from scipy.special import gamma as Gamma

from WeOptPy.algorithms.interfaces import Algorithm
from WeOptPy.util import reflect_repair

__all__ = ['FlowerPollinationAlgorithm']


class FlowerPollinationAlgorithm(Algorithm):
	r"""Implementation of Flower Pollination algorithm.

	Algorithm:
		Flower Pollination algorithm

	Date:
		2018

	Authors:
		Dusan Fister, Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Yang, Xin-She. "Flower pollination algorithm for global optimization. International conference on unconventional computing and natural computation. Springer, Berlin, Heidelberg, 2012.

	References URL:
		Implementation is based on the following MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/45112-flower-pollination-algorithm?requestedDomain=true

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		p (float): probability switch.
		beta (float): Shape of the gamma distribution (should be greater than zero).

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['FlowerPollinationAlgorithm', 'FPA']

	@staticmethod
	def type_parameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* p (function): TODO
				* beta (function): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'p': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'beta': lambda x: isinstance(x, (float, int)) and x > 0,
		})
		return d

	def set_parameters(self, n=25, p=0.35, beta=1.5, **ukwargs):
		r"""Set core parameters of FlowerPollinationAlgorithm algorithm.

		Args:
			n (int): Population size.
			p (float): Probability switch.
			beta (float): Shape of the gamma distribution (should be greater than zero).

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, **ukwargs)
		self.p, self.beta = p, beta
		self.S = np.zeros((n, 10))

	def levy(self, D):
		r"""Levy function.

		Returns:
			float: Next Levy number.
		"""
		sigma = (Gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / (Gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
		return 0.01 * (self.normal(0, 1, D) * sigma / np.fabs(self.normal(0, 1, D)) ** (1 / self.beta))

	def init_population(self, task):
		r"""Initialize the initial population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. Initial population.
				2. Initials population fitness/utility function values.
				3. Additional arguments.
				4. Additional keyword arguments.
		"""
		pop, fpop, args, d = Algorithm.init_population(self, task)
		d.update({'S': np.zeros((self.NP, task.D))})
		return pop, fpop, args, d

	def run_iteration(self, task, Sol, Sol_f, xb, fxb, S, *args, **dparams):
		r"""Core function of FlowerPollinationAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population.
			Sol_f (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solution function/fitness value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best solution
				4. New global best solution fitness/objective value
				5. Additional arguments.
				6. Additional keyword arguments.
		"""
		for i in range(self.NP):
			if self.uniform(0, 1) > self.p: S[i] += self.levy(task.D) * (Sol[i] - xb)
			else:
				JK = self.Rand.permutation(self.NP)
				S[i] += self.uniform(0, 1) * (Sol[JK[0]] - Sol[JK[1]])
			S[i] = reflect_repair(S[i], task.Lower, task.Upper)
			f_i = task.eval(S[i])
			if f_i <= Sol_f[i]: Sol[i], Sol_f[i] = S[i], f_i
			if f_i <= fxb: xb, fxb = S[i].copy(), f_i
		return Sol, Sol_f, xb, fxb, args, {'S': S}


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
