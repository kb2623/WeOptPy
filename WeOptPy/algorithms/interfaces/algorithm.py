# encoding=utf8

"""Algorithm interface module."""

import logging

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

logging.basicConfig()
logger = logging.getLogger('WeOptPy.test')
logger.setLevel('INFO')

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
		NP (int): Number of individuals in population.
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
			* :func:`WeOptPy.algorithms.interfaces.Algorithm.setParameters`
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
			* :func:`WeOptPy.algorithms.defaultNumPyInit`
			* :func:`WeOptPy.algorithms.defaultIndividualInit`
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
		r"""Get uniform random distribution of shape d in range from "lower" to "upper".

		Args:
			lower (Union[float, numpy.ndarray]): lower bound.
			upper (Union[float, numpy.ndarray]): upper bound.
			d (Optional[Union[int, Iterable[int]]]): Shape of returned uniform random distribution.

		Returns:
			Union[float, numpy.ndarray]: Array of numbers :math:`\in [\mathit{lower}, \mathit{upper}]`.
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

	def randint(self, maximum, d=1, minimum=0, skip=None):
		r"""Get discrete uniform (integer) random distribution of d shape in range from "minimum" to "Nmax".

		Args:
			minimum (int): lower integer bound.
			d (Optional[Union[int, Iterable[int]]]): shape of returned discrete uniform random distribution.
			maximum (Optional[int]): One above upper integer bound.
			skip (Optional[Union[int, Iterable[int]]]): numbers to skip.

		Returns:
			Union[int, numpy.ndarray]: Random generated integer number.
		"""
		r = None
		if isinstance(d, (list, tuple, np.ndarray)): r = self.Rand.randint(minimum, maximum, d)
		elif d > 1: r = self.Rand.randint(minimum, maximum, d)
		else: r = self.Rand.randint(minimum, maximum)
		return r if skip is None or r not in skip else self.randint(maximum, d, minimum, skip)

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
			Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
				1. New population.
				2. New population fitness values.
				3. Additional arguments.
				4. Additional keyword arguments.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.setParameters`
		"""
		pop, fpop, args, kwargs = self.InitPopFunc(task=task, n=self.NP, rnd=self.Rand, itype=self.itype)
		return pop, fpop, args, kwargs

	def run_iteration(self, task, pop, fpop, xb, fxb, *args, **dparams):
		r"""Core functionality of algorithm.

		This function is called on every algorithm iteration.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population coordinates.
			fpop (numpy.ndarray): Current population fitness value.
			xb (numpy.ndarray): Current generation best individuals coordinates.
			fxb (float): current generation best individuals fitness value.
			args (list): Additional arguments.
			dparams (dict): Additional keyword arguments for algorithms.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
				1. New populations coordinates.
				2. New populations fitness values.
				3. New global best position/solution.
				4. New global best fitness/objective value.
				5. Additional arguments.
				6. Additional keyword arguments of the algorithm.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.runYield`
		"""
		return pop, fpop, xb, fxb, args, dparams

	def run_yield(self, task, *args, **kwargs):
		r"""Run the algorithm for a single iteration and return the best solution.

		Args:
			task (Task): Task with bounds and objective function for optimization.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Keyword Args:
			store (bool): Store algorithm state to files.
			store_file_name (str): Store file name.
			log (bool): Log current state to output.

		Returns:
			Generator[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]: Generator getting new/old optimal global values.

		Yield:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. New population best individuals coordinates.
				2. Fitness value of the best solution.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.initPopulation`
			* :func:`WeOptPy.algorithms.Algorithm.runIteration`
		"""
		# TODO add caching to algorithms
		"""
		* if caching no set then just as is
		* if user set caching then
			* check if cache exists on disk
				* if cache exists 
					* load population and all parameters from dist
				* if no cache exists
					* create new population and all arguments
					* cache the population and all arguments
				* start optimization
				* while optimizing save population and all parameters
		"""
		log_b, store_b = kwargs.get('log', False), kwargs.get('store', False)
		pop, fpop, a_args, a_kwargs = self.init_population(task)
		xb, fxb = self.get_best(pop, fpop)
		if log_b: logger.info('%s -> %s' % (xb, fxb))
		yield xb, fxb
		while not task.stop_cond():
			pop, fpop, xb, fxb, a_args, a_kwargs = self.run_iteration(task, pop, fpop, xb, fxb, *a_args, **a_kwargs)
			if log_b: logger.info('%s -> %s' % (xb, fxb))
			# TODO store procedure
			yield xb, fxb
		return xb, fxb

	def run_task(self, task, *args, **kwargs):
		r"""Start the optimization.

		Args:
			task (Task): Task with bounds and objective function for optimization.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.runYield`
		"""
		xb, fxb = None, np.inf
		for x, fx in self.run_yield(task, *args, **kwargs):
			xb, fxb = x, fx
			task.next_iteration()
		return xb, fxb

	def run(self, task, *args, **kwargs):
		r"""Start the optimization.

		Args:
			task (Task): Optimization task.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.runTask`
		"""
		try:
			task.start()
			r = self.run_task(task, *args, **kwargs)
			return r[0], r[1] * task.optType.value
		except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value
		except Exception as e: self.exception = e
		return None, None

	def __call__(self, task, *args, **kwargs):
		r"""Start the optimization.

		Args:
			task (Task): Optimization task.
			args (list): Additional arguments.
			kwargs (dict): Additional keyword arguments.
			
		Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.

		See Also:
			* :func:`WeOptPy.algorithms.Algorithm.run`
		"""
		return self.run(task, *args, **kwargs)

	def bad_run(self):
		r"""Check if some exceptions where thrown when the algorithm was running.

		Returns:
			bool: True if some error where detected at runtime of the algorithm, otherwise False
		"""
		return self.exception is not None


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
