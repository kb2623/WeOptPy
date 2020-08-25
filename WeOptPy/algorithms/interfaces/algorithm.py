# encoding=utf8

import numpy as np
from numpy import random as rand

from WeOptPy.algorithms.interfaces.individual import (
	Individual,
	default_numpy_init
)
from WeOptPy.util.exception import (
	FesException,
	GenException,
	TimeException,
	RefException
)

__all__ = ['Algorithm']


class Algorithm:
	r"""Class for implementing algorithms.

	Date:
		2018

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of names for algorithm.
		Rand (rand.RandomState): Random generator.
		NP (int): Number of inidividuals in populatin.
		InitPopFunc (Callable[[Task, int, Optional[rand.RandomState], Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]): Idividual initialization function.
		itype (Individual): Type of individuals used in population, default value is None for Numpy arrays.
	"""
	Name = ['Algorithm', 'AAA']
	Rand = rand.RandomState(None)
	NP = 50
	InitPopFunc = default_numpy_init
	itype = None

	@staticmethod
	def type_parameters():
		r"""Return functions for checking values of parameters.

		Return:
			Dict[str, Callable[[Any], bool]]:
				* n: Check if number of individuals is :math:`\in [0, \infty]`.
		"""
		return {'n': lambda x: isinstance(x, int) and x > 0}

	def __init__(self, seed=None, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		Args:
			seed (Optional[int]): Starting seed for random generator.
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		self.Rand, self.exception = rand.RandomState(seed), None
		self.set_parameters(**kwargs)

	@staticmethod
	def algorithm_info():
		r"""Get algorithm information.

		Returns:
			str: Bit item.
		"""
		return '''Basic algorithm. No implementation!!!'''

	def set_parameters(self, n=50, init_pop_func=default_numpy_init, itype=None, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		Args:
			n (Optional[int]): Number of individuals in population :math:`\in [1, \infty]`.
			init_pop_func (Optional[Callable[[Task, int, Optional[rand.RandomState], Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]]): Type of individuals used by algorithm.
			itype (Individual): Individual type used in population, default is Numpy array.
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.defaultNumPyInit`
			* :func:`NiaPy.algorithms.defaultIndividualInit`
		"""
		self.NP, self.InitPopFunc, self.itype = n, init_pop_func, itype

	def get_parameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
				* Parameter name: Represents a parameter name
				* Value of parameter: Represents the value of the parameter
		"""
		return {
			'n': self.NP,
			'init_pop_func': self.InitPopFunc,
			'itype': self.itype
		}

	def rand(self, d=1):
		r"""Get random distribution of shape d in range from 0 to 1.

		Args:
			d (Optional[int]): Shape of returned random distribution.

		Returns:
			Union[float, numpy.ndarray]: Random number or numbers :math:`\in [0, 1]`.
		"""
		if isinstance(d, (np.ndarray, list)): return self.Rand.rand(*d)
		elif d > 1: return self.Rand.rand(d)
		else: return self.Rand.rand()

	def uniform(self, lower, upper, d=None):
		r"""Get uniform random distribution of shape d in range from "Lower" to "Upper".

		Args:
			lower (Union[float, numpy.ndarray]): Lower bound.
			upper (Union[float, numpy.ndarray]): Upper bound.
			d (Optional[Union[int, Iterable[int]]]): Shape of returned uniform random distribution.

		Returns:
			Union[float, numpy.ndarray]: Array of numbers :math:`\in [\mathit{Lower}, \mathit{Upper}]`.
		"""
		return self.Rand.uniform(lower, upper, d) if d is not None else self.Rand.uniform(lower, upper)

	def normal(self, loc, scale, d=None):
		r"""Get normal random distribution of shape d with mean "loc" and standard deviation "scale".

		Args:
			loc (float): Mean of the normal random distribution.
			scale (float): Standard deviation of the normal random distribution.
			d (Optional[Union[int, Iterable[int]]]): Shape of returned normal random distribution.

		Returns:
			Union[numpy.ndarray, float]: Array of numbers.
		"""
		return self.Rand.normal(loc, scale, d) if d is not None else self.Rand.normal(loc, scale)

	def randn(self, d=None):
		r"""Get standard normal distribution of shape d.

		Args:
			d (Optional[Union[int, Iterable[int]]]): Shape of returned standard normal distribution.

		Returns:
			Union[numpy.ndarray, float]: Random generated numbers or one random generated number :math:`\in [0, 1]`.
		"""
		if d is None: return self.Rand.randn()
		elif isinstance(d, int): return self.Rand.randn(d)
		return self.Rand.randn(*d)

	def randint(self, nmax, d=1, nmin=0, skip=None):
		r"""Get discrete uniform (integer) random distribution of d shape in range from "nmin" to "Nmax".

		Args:
			nmin (int): lower integer bound.
			d (Optional[Union[int, Iterable[int]]]): shape of returned discrete uniform random distribution.
			nmax (Optional[int]): One above upper integer bound.
			skip (Optional[Union[int, Iterable[int]]]): numbers to skip.

		Returns:
			Union[int, numpy.ndarray]: Random generated integer number.
		"""
		r = None
		if isinstance(d, (list, tuple, np.ndarray)): r = self.Rand.randint(nmin, nmax, d)
		elif d > 1: r = self.Rand.randint(nmin, nmax, d)
		else: r = self.Rand.randint(nmin, nmax)
		return r if skip is None or r not in skip else self.randint(nmax, d, nmin, skip)

	def get_best(self, x, x_f, xb=None, xb_f=np.inf):
		r"""Get the best individual for population.

		Args:
			x (numpy.ndarray): Current population.
			x_f (numpy.ndarray): Current populations fitness/function values of aligned individuals.
			xb (Optional[numpy.ndarray]): Best individual.
			xb_f (Optional[float]): Fitness value of best individual.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Coordinates of best solution.
				2. beset fitness/function value.
		"""
		ib = np.argmin(x_f)
		if isinstance(x_f, (float, int)) and xb_f >= x_f: xb, xb_f = x, x_f
		elif isinstance(x_f, (np.ndarray, list)) and xb_f >= x_f[ib]: xb, xb_f = x[ib], x_f[ib]
		return (xb.x.copy() if isinstance(xb, Individual) else xb.copy()), xb_f

	def init_population(self, task):
		r"""Initialize starting population of optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
				1. New population.
				2. New population fitness values.
				3. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		pop, fpop = self.InitPopFunc(task=task, NP=self.NP, rnd=self.Rand, itype=self.itype)
		return pop, fpop, {}

	def run_iteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core functionality of algorithm.

		This function is called on every algorithm iteration.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population coordinates.
			fpop (numpy.ndarray): Current population fitness value.
			xb (numpy.ndarray): Current generation best individuals coordinates.
			fxb (float): current generation best individuals fitness value.
			dparams (Dict[str, Any]): Additional arguments for algorithms.

		Returns:
			Tuple[n.ndarray, n.ndarray, n.ndarray, float, Dict[str, Any]]:
				1. New populations coordinates.
				2. New populations fitness values.
				3. New global best position/solution
				4. New global best fitness/objective value
				5. Additional arguments of the algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runYield`
		"""
		return pop, fpop, xb, fxb, dparams

	def run_yield(self, task):
		r"""Run the algorithm for a single iteration and return the best solution.

		Args:
			task (Task): Task with bounds and objective function for optimization.

		Returns:
			Generator[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray], None]: Generator getting new/old optimal global values.

		Yield:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. New population best individuals coordinates.
				2. Fitness value of the best solution.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
			* :func:`NiaPy.algorithms.Algorithm.runIteration`
		"""
		pop, fpop, dparams = self.init_population(task)
		xb, fxb = self.get_best(pop, fpop)
		yield xb, fxb
		while True:
			pop, fpop, xb, fxb, dparams = self.run_iteration(task, pop, fpop, xb, fxb, **dparams)
			yield xb, fxb

	def run_task(self, task):
		r"""Start the optimization.

		Args:
			task (Task): Task with bounds and objective function for optimization.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runYield`
		"""
		algo, xb, fxb = self.run_yield(task), None, np.inf
		while not task.stop_cond():
			xb, fxb = next(algo)
			task.nextIter()
		return xb, fxb

	def run(self, task):
		r"""Start the optimization.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runTask`
		"""
		try:
			# task.start()
			r = self.run_task(task)
			return r[0], r[1] * task.optType.value
		except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value
		except Exception as e: self.exception = e
		return None, None

	def __call__(self, task):
		r"""Start the optimization.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.run`
		"""
		return self.run(task)

	def bad_run(self):
		r"""Check if some exceptions where thrown when the algorithm was running.

		Returns:
			bool: True if some error where detected at runtime of the algorithm, otherwise False
		"""
		return self.exception is not None


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
