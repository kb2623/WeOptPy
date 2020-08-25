# encoding=utf8

"""The implementation of tasks."""

import logging

import numpy as np
from matplotlib import pyplot as plt

from WeOptPy.task.countingtask import CountingTask

logging.basicConfig()
logger = logging.getLogger('NiaPy.runner.Runner')
logger.setLevel('INFO')


class StoppingTask(CountingTask):
	r"""Optimization task with implemented checking for stopping criterias.

	Args:
		no_gen (int): Maximum number of algorithm iterations/generations.
		no_fes (int): Maximum number of function evaluations.
		rvalue (float): Reference function/fitness values to reach in optimization.
		x (numpy.ndarray): Best found individual.
		x_f (float): Best found individual function/fitness value.

	See Also:
		* :class:`NiaPy.util.CountingTask`
	"""
	def __init__(self, no_fes=np.inf, no_gen=np.inf, rvalue=None, verbose=False, **kwargs):
		r"""Initialize task class for optimization.

		Arguments:
			no_fes (Optional[int]): Number of function evaluations.
			no_gen (Optional[int]): Number of generations or iterations.
			rvalue (Optional[float]): Reference value of function/fitness function.
			verbose (Optional[bool]): Enable/disable logging of improvements.

		Note:
			Storing improvements during the evolutionary cycle is
			captured in self.n_evals and self.x_f_vals

		See Also:
			* :func:`NiaPy.util.CountingTask.__init__`
		"""
		CountingTask.__init__(self, **kwargs)
		self.refValue = (-np.inf if rvalue is None else rvalue)
		self.verbose = verbose
		self.x, self.x_f = None, np.inf
		self.nFES, self.nGEN = no_fes, no_gen
		self.n_evals = []
		self.x_f_vals = []

	def eval(self, a):
		r"""Evaluate solution.

		Args:
			a (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Fitness/function value of solution.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.CountingTask.eval`
		"""
		if self.stop_cond(): return np.inf * self.optType.value
		x_f = CountingTask.eval(self, a)
		if x_f < self.x_f:
			self.x_f = x_f
			self.n_evals.append(self.Evals)
			self.x_f_vals.append(x_f)
			if self.verbose: logger.info('nFES:%d => %s' % (self.Evals, self.x_f))
		return x_f

	def stop_cond(self):
		r"""Check if stopping condition reached.

		Returns:
			bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`
		"""
		return (self.Evals >= self.nFES) or (self.Iters >= self.nGEN) or (self.refValue > self.x_f)

	def stop_cond_i(self):
		r"""Check if stopping condition reached and increase number of iterations.

		Returns:
			bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.CountingTask.nextIter`
		"""
		r = self.stop_cond()
		CountingTask.nextIter(self)
		return r

	def return_conv(self):
		r"""Get values of x and y axis for plotting covariance graph.

		Returns:
			Tuple[List[int], List[float]]:
				1. List of ints of function evaluations.
				2. List of ints of function/fitness values.
		"""
		r1, r2 = [], []
		for i, v in enumerate(self.n_evals):
			r1.append(v), r2.append(self.x_f_vals[i])
			if i >= len(self.n_evals) - 1: break
			diff = self.n_evals[i + 1] - v
			if diff <= 1: continue
			for j in range(diff - 1): r1.append(v + j + 1), r2.append(self.x_f_vals[i])
		return r1, r2

	def plot(self):
		"""Plot a simple convergence graph."""
		fess, fitnesses = self.return_conv()
		plt.plot(fess, fitnesses)
		plt.xlabel('nFes')
		plt.ylabel('Fitness')
		plt.title('Convergence graph')
		plt.show()


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
