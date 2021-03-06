# encoding=utf8

"""Nelder mead method algorithm module."""

import numpy as np

from WeOptPy.algorithms.interfaces import Algorithm

__all__ = ['NelderMeadMethod']


class NelderMeadMethod(Algorithm):
	r"""Implementation of Nelder Mead method or downhill simplex method or amoeba method.

	Algorithm:
		Nelder Mead Method

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

	Attributes:
		Name (List[str]): list of strings represeing algorithm name
		alpha (float): Reflection coefficient parameter
		gamma (float): Expansion coefficient parameter
		rho (float): Contraction coefficient parameter
		sigma (float): Shrink coefficient parameter

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['NelderMeadMethod', 'NMM']

	@staticmethod
	def algorithm_info():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""No info"""

	@staticmethod
	def type_parameters():
		r"""Get dictionary with function for testing correctness of parameters.

		Returns:
			Dict[str, Callable]:
				* alpha (Callable[[Union[int, float]], bool])
				* gamma (Callable[[Union[int, float]], bool])
				* rho (Callable[[Union[int, float]], bool])
				* sigma (Callable[[Union[int, float]], bool])

		See Also
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.type_parameters()
		d.update({
			'alpha': lambda x: isinstance(x, (int, float)) and x >= 0,
			'gamma': lambda x: isinstance(x, (int, float)) and x >= 0,
			'rho': lambda x: isinstance(x, (int, float)),
			'sigma': lambda x: isinstance(x, (int, float))
		})
		return d

	def set_parameters(self, n=None, alpha=0.1, gamma=0.3, rho=-0.2, sigma=-0.2, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			n (Optional[int]): Number of individuals.
			alpha (Optional[float]): Reflection coefficient parameter
			gamma (Optional[float]): Expansion coefficient parameter
			rho (Optional[float]): Contraction coefficient parameter
			sigma (Optional[float]): Shrink coefficient parameter

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.set_parameters(self, n=n, init_pop_func=ukwargs.pop('init_pop_func', self.init_pop), **ukwargs)
		self.alpha, self.gamma, self.rho, self.sigma = alpha, gamma, rho, sigma

	def get_parameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
				* Parameter name: Represents a parameter name
				* Value of parameter: Represents the value of the parameter
		"""
		d = Algorithm.get_parameters(self)
		d.update({
			'alpha': self.alpha,
			'gamma': self.gamma,
			'rho': self.rho,
			'sigma': self.sigma
		})
		return d

	def init_pop(self, task, n, *args, **kwargs):
		r"""Init starting population.

		Args:
			task (Task): Optimization task.
			n (int): Number of individuals in population.
			rnd (mtrand.RandomState): Random number generator.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New initialized population.
				2. New initialized population fitness/function values.
				3. Additional arguments.
				4. Additional keyword arguments.
		"""
		X = self.uniform(task.Lower, task.Upper, [task.D if n is None or n < task.D else n, task.D])
		X_f = np.apply_along_axis(task.eval, 1, X)
		return X, X_f, args, kwargs

	def method(self, X, X_f, task):
		r"""Run the main function.

		Args:
			X (numpy.ndarray): Current population.
			X_f (numpy.ndarray[float]): Current population function/fitness values.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float]]:
				1. New population.
				2. New population fitness/function values.
		"""
		x0 = np.sum(X[:-1], axis=0) / (len(X) - 1)
		xr = x0 + self.alpha * (x0 - X[-1])
		rs = task.eval(xr)
		if X_f[0] >= rs < X_f[-2]:
			X[-1], X_f[-1] = xr, rs
			return X, X_f
		if rs < X_f[0]:
			xe = x0 + self.gamma * (x0 - X[-1])
			re = task.eval(xe)
			if re < rs: X[-1], X_f[-1] = xe, re
			else: X[-1], X_f[-1] = xr, rs
			return X, X_f
		xc = x0 + self.rho * (x0 - X[-1])
		rc = task.eval(xc)
		if rc < X_f[-1]:
			X[-1], X_f[-1] = xc, rc
			return X, X_f
		Xn = X[0] + self.sigma * (X[1:] - X[0])
		Xn_f = np.apply_along_axis(task.eval, 1, Xn)
		X[1:], X_f[1:] = Xn, Xn_f
		return X, X_f

	def run_iteration(self, task, X, X_f, xb, fxb, *args, **dparams):
		r"""Core iteration function of NelderMeadMethod algorithm.

		Args:
			task (Task): Optimization task.
			X (numpy.ndarray): Current population.
			X_f (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best function/fitness value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New population.
				2. New population fitness/function values.
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Additional arguments.
				6. Additional keywrod arguments.
		"""
		inds = np.argsort(X_f)
		X, X_f = X[inds], X_f[inds]
		X, X_f = self.method(X, X_f, task)
		xb, fxb = self.get_best(X, X_f, xb, fxb)
		return X, X_f, xb, fxb, args, dparams


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
